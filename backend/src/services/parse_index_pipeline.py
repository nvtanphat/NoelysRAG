from __future__ import annotations

import gc
import json
import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from beanie import PydanticObjectId

from src.core.config import Settings
from src.core.model_factory import build_llm
from src.models.common import PipelineStatus, utc_now
from src.models.collection import KnowledgeCollection
from src.models.material import Material, replace_material_pages
from src.models.pipeline_job import PipelineJob
from src.processing.audio_parser import AUDIO_EXTENSIONS, AudioParser
from src.processing.chunking import AudioChunker, LayoutAwareChunker, SemanticChunker, SlideAwareChunker
from src.processing.contextual_enricher import ContextualEnricher
from src.processing.docling_parser import DoclingParser, SUPPORTED_DOCLING_EXTENSIONS
from src.processing.cross_modal_linker import CrossModalLinker
from src.processing.structure_linker import StructureLinker
from src.processing.entity_extractor import EntityExtractor
from src.processing.section_filter import filter_extraction_blocks
from src.processing.entity_resolution import EntityResolver
from src.processing.semantic_relation_extractor import LLMSemanticRelationExtractor
from src.processing.event_extractor import EventExtractor
from src.processing.evidence_mapper import EvidenceMapper
from src.processing.graph_quality_gate import GraphQualityGate
from src.processing.handwriting_reader import HandwritingReader
from src.processing.layout_normalizer import LayoutNormalizer
from src.processing.language_detector import detect_block_language, detect_document_language
from src.processing.chunk_qa import run_chunk_qa
from src.processing.figure_captioner import FigureCaptioner
from src.processing.ocr_engine import EasyOCREngine, VietOCRRecognizer
from src.processing.ocr_quality_gate import score_ocr_document
from src.processing.spreadsheet_parser import SpreadsheetParser, SUPPORTED_SPREADSHEET_EXTENSIONS
from src.processing.types import BlockType, OCRQualityError, ParsedBlock, ParsedDocument, ParsedPage
from src.rag.indexer import QdrantMongoIndexer
from src.rag.types import FigureIndexItem


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

# Heuristic threshold for "figure block is really the whole page" detection.
# Docling marks scanned-text PDF pages as a single FIGURE block with a bbox
# covering most of the page; we want full-page OCR in that case, not a crop.
_FULL_PAGE_AREA_RATIO = 0.7


def _bbox_covers_most_of_page(bbox, page_width: int, page_height: int) -> bool:
    """True when the bbox area exceeds _FULL_PAGE_AREA_RATIO of the page area.

    Accepts either a BBox dataclass / pydantic model (with x/y/width/height
    attrs) or a dict shaped like {"x":..., "y":..., "width":..., "height":...}.
    Returns False on any malformed input so the caller falls back to the
    normal crop path.
    """
    if bbox is None or page_width <= 0 or page_height <= 0:
        return False
    width = getattr(bbox, "width", None) or (bbox.get("width") if isinstance(bbox, dict) else None)
    height = getattr(bbox, "height", None) or (bbox.get("height") if isinstance(bbox, dict) else None)
    if not width or not height:
        return False
    try:
        area_ratio = (float(width) * float(height)) / (float(page_width) * float(page_height))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return area_ratio >= _FULL_PAGE_AREA_RATIO


def _table_has_numeric_data(caption: str, *, min_digit_cell_ratio: float) -> bool:
    """True when the VLM caption's markdown table(s) carry real numeric data.

    The VLM structure pass only runs on numeric-dense images, so a "table" whose
    data columns are empty is a failure (small VLMs collapse multi-panel pages
    into a hollow grid: only the year/label column is filled, every value column
    blank). Counting digit cells overall does NOT catch this — the year column is
    itself numeric. Instead we count *numeric columns*: a body column where at
    least ``min_digit_cell_ratio`` of its cells contain a digit. A genuine
    chart/table has >= 2 numeric columns (year + >=1 value series, in either
    orientation); the hollow bug has exactly one (the year/label column).

    A prose caption (no markdown table parses) is accepted unchanged — only
    dataless tables are rejected. Multi-section captions (the prompt emits
    "## region" tables) qualify if ANY section is a real data table.
    """
    from src.processing.table_serializer import parse_markdown_table

    sections = re.split(r"(?m)^##\s+.*$", caption or "")
    if len(sections) <= 1:
        sections = [caption or ""]

    parsed_any = False
    for section in sections:
        parsed = parse_markdown_table(section)
        if parsed is None:
            continue
        parsed_any = True
        _header, rows = parsed
        if not rows:
            continue
        width = max(len(r) for r in rows)
        numeric_columns = 0
        for col in range(width):
            cells = [r[col] for r in rows if col < len(r)]
            if not cells:
                continue
            digit_cells = sum(1 for c in cells if any(ch.isdigit() for ch in c))
            if (digit_cells / len(cells)) >= min_digit_cell_ratio:
                numeric_columns += 1
        if numeric_columns >= 2:
            return True

    # No section parsed as a table → prose caption, not the bug → accept.
    # Otherwise every parsed table was hollow (< 2 numeric columns) → reject.
    return not parsed_any


class ParseIndexPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        docling_parser: DoclingParser | None = None,
        ocr_engine: EasyOCREngine | None = None,
        spreadsheet_parser: SpreadsheetParser | None = None,
        handwriting_reader: HandwritingReader | None = None,
        normalizer: LayoutNormalizer | None = None,
        evidence_mapper: EvidenceMapper | None = None,
        entity_extractor: EntityExtractor | None = None,
        entity_resolver: EntityResolver | None = None,
        event_extractor: EventExtractor | None = None,
        cross_modal_linker: CrossModalLinker | None = None,
        structure_linker: StructureLinker | None = None,
        graph_quality_gate: GraphQualityGate | None = None,
        chunker: LayoutAwareChunker | SemanticChunker | None = None,
        indexer: QdrantMongoIndexer | None = None,
    ) -> None:
        self.settings = settings
        self.docling_parser = docling_parser or DoclingParser()
        self.ocr_engine = ocr_engine
        self._ocr_engines: dict[str, EasyOCREngine] = {}
        self.spreadsheet_parser = spreadsheet_parser or SpreadsheetParser()
        self.handwriting_reader = handwriting_reader or HandwritingReader(settings=settings)
        self.audio_parser = AudioParser(settings=settings)
        self.normalizer = normalizer or LayoutNormalizer(
            column_detection_enabled=settings.layout_column_detection_enabled,
            column_min_gutter_ratio=settings.layout_column_min_gutter_ratio,
            column_min_band_occupancy_ratio=settings.layout_column_min_band_occupancy_ratio,
        )
        self.evidence_mapper = evidence_mapper or EvidenceMapper()
        _llm = build_llm(settings)
        self.entity_extractor = entity_extractor or EntityExtractor(
            llm=_llm,
            default_entity_types=settings.extraction_default_entity_types,
            few_shots=settings.extraction_few_shots,
            mode=settings.extraction_mode,
        )
        self.entity_resolver = entity_resolver or EntityResolver()
        self.event_extractor = event_extractor or EventExtractor()
        self.cross_modal_linker = cross_modal_linker or CrossModalLinker()
        self.structure_linker = structure_linker or StructureLinker(
            enabled=settings.structure_linker_enabled,
            section_confidence=settings.structure_section_confidence,
            belongs_to_confidence=settings.structure_belongs_to_confidence,
        )
        # Semantic relation extraction + entity refinement add LLM calls per
        # material on top of entity extraction. Cloud routing removed — these
        # now run on the same local LLM as answer generation (qwen2.5:7b). On a
        # GPU host this is reliable; on a tiny local model (e.g. qwen2.5:3b) the
        # Ollama queue can saturate, so use a capable local model.
        _semantic_llm = _llm
        self.semantic_relation_extractor = LLMSemanticRelationExtractor(
            llm=_semantic_llm,
            max_concepts=settings.graph_semantic_relation_max_concepts,
            max_passages=settings.graph_semantic_relation_max_passages,
            max_passage_chars=settings.graph_semantic_relation_max_passage_chars,
            gleaning=settings.graph_semantic_relation_gleaning,
        )
        # Ontology-guided entity refinement shares the same cloud LLM as semantic
        # relations (capable model required for de-fragmentation / re-typing).
        if settings.extraction_entity_refine_enabled and _semantic_llm is not None:
            from src.processing.entity_refiner import EntityRefiner

            self.entity_refiner = EntityRefiner(
                llm=_semantic_llm,
                allowed_types=tuple(settings.extraction_graph_entity_types),
                type_descriptions=settings.extraction_entity_type_descriptions,
                max_entities=settings.extraction_entity_refine_max_entities,
            )
        else:
            self.entity_refiner = None
        self.graph_quality_gate = graph_quality_gate or GraphQualityGate(
            min_entity_confidence=settings.min_graph_confidence,
            min_relation_confidence=settings.min_graph_confidence,
            min_mention_count=1,
        )
        self._chunker_override = chunker
        self.indexer = indexer

    async def run(self, *, material_id: str, job_id: str) -> None:
        material = await Material.get(PydanticObjectId(material_id))
        if material is None:
            raise LookupError(f"Material not found: {material_id}")
        job = await PipelineJob.find_one(PipelineJob.job_id == job_id)
        if job is None:
            raise LookupError(f"Pipeline job not found: {job_id}")

        try:
            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.PARSING.value, stage=PipelineStatus.PARSING.value)
            parsed = await asyncio.to_thread(self._parse_material, material)
            parsed = await asyncio.to_thread(self._caption_figures, parsed, material.language)
            parsed = await asyncio.to_thread(self._vlm_table_fallback, parsed, material.language)
            # Collect figure items for visual indexing before normalization may
            # drop block.extra fields.  Empty when visual embedding is disabled.
            figure_items = (
                self._collect_figure_items(parsed, material)
                if self.settings.visual_embedding_enabled
                else []
            )
            normalized = await asyncio.to_thread(self.normalizer.normalize, parsed)
            detected_language, language_counts = self._apply_language_detection(normalized, declared_language=material.language)
            material.extra_metadata["detected_language"] = detected_language
            material.extra_metadata["detected_language_counts"] = language_counts
            if material.language == "unknown" and detected_language != "unknown":
                material.language = detected_language
            material_pages = self.normalizer.to_material_pages(normalized)
            await replace_material_pages(material, material_pages)
            material.pages = []
            material.page_count = len(material_pages)
            await self._write_processed_artifacts(material, normalized)
            await self._mark(material=material, job=job, status=PipelineStatus.PARSED.value, stage=PipelineStatus.PARSED.value)
            gc.collect()  # free docling/OCR memory before loading embedding model

            await self._ensure_material_exists(material_id)
            # Initialise indexer early so SemanticChunker can reuse its embedder
            if self.indexer is None:
                from src.dependencies import get_qdrant_client

                self.indexer = QdrantMongoIndexer(settings=self.settings, qdrant_client=get_qdrant_client())

            await self._mark(material=material, job=job, status=PipelineStatus.CHUNKING.value, stage=PipelineStatus.CHUNKING.value)
            evidence_map = self.evidence_mapper.build(
                parsed=normalized,
                owner_id=material.owner_id,
                collection_id=str(material.collection_id),
                material_id=str(material.id),
                document_name=material.original_name,
            )
            chunker = self._resolve_chunker(file_type=material.file_type)
            if isinstance(chunker, SemanticChunker):
                chunks = await chunker.build_chunks_async(evidence_map)
            else:
                chunks = chunker.build_chunks(evidence_map)
            chunks = [c for c in chunks if len((c.content or "").strip()) >= self.settings.chunk_min_content_chars]
            run_chunk_qa(chunks, material_id=str(material.id))
            domain_hint = await self._fetch_domain_hint(material)
            # Section-aware extraction: drop References/Acknowledgments/etc from the
            # graph input only. Chunking/retrieval above still see the full document.
            graph_map = filter_extraction_blocks(
                evidence_map,
                excluded_patterns=tuple(self.settings.extraction_excluded_sections) or None,
            )
            raw_entities = await self.entity_extractor.extract_async(graph_map, domain_hint=domain_hint)
            # Ontology-guided LLM refinement: fix GLiNER mistypes / fragmented
            # spans on Vietnamese, drop junk, glean misses. No-op when disabled
            # or no cloud LLM is configured. Runs on graph_map (extraction scope).
            if self.entity_refiner is not None:
                try:
                    raw_entities = await self.entity_refiner.refine_async(raw_entities, graph_map)
                except Exception as exc:
                    logger.warning(
                        "Entity refinement failed (non-fatal); using raw entities",
                        extra={"error": str(exc), "error_type": type(exc).__name__},
                    )
            embedder = self.indexer.embedder if self.indexer is not None else None
            entities = await self.entity_resolver.resolve_async(raw_entities, embedder=embedder)
            events, relations = self.event_extractor.extract(graph_map, entities)
            cm_entities, cm_relations = self.cross_modal_linker.link(evidence_map, entities)
            # Structural hierarchy: section entities + belongs_to edges (positional, no ML).
            struct_entities, struct_relations = self.structure_linker.link(evidence_map)
            # LLM-driven typed semantic relations between concept entities
            # (uses / extends / improves / compared_with / …). Empty when
            # the LLM is unavailable or the doc has < 2 concept entities.
            try:
                semantic_relations = await self.semantic_relation_extractor.extract_async(
                    evidence_map, entities,
                )
            except Exception as exc:
                logger.warning(
                    "Semantic relation extraction failed (non-fatal)",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
                semantic_relations = []
            entities = list(entities) + cm_entities + struct_entities
            relations = list(relations) + cm_relations + semantic_relations + struct_relations

            # Apply graph quality gates
            entities = self.graph_quality_gate.prune_entities(entities)
            entities = self.graph_quality_gate.resolve_entities(entities)

            # Build valid entity ID set for relation pruning
            valid_entity_ids = {
                f"entity:{self.graph_quality_gate._slug(e.canonical_name)}"
                for e in entities
            }
            # Also include event IDs and block IDs
            for event in events:
                valid_entity_ids.add(f"event:{self.graph_quality_gate._slug(event.event_name)}")
            for block in evidence_map.blocks:
                valid_entity_ids.add(f"block:{block.block_id}")

            relations = self.graph_quality_gate.prune_relations(relations, valid_entity_ids)

            logger.info(
                "Graph extraction completed",
                extra={
                    "entities": len(entities),
                    "events": len(events),
                    "relations": len(relations),
                    "material_id": str(material.id),
                },
            )

            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.EMBEDDING.value, stage=PipelineStatus.EMBEDDING.value)
            chunks = await self._contextual_enrich(chunks, evidence_map)

            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.INDEXING.value, stage=PipelineStatus.INDEXING.value)
            await self.indexer.index(
                chunks=chunks,
                entities=entities,
                events=events,
                relations=relations,
                material_id=str(material.id),
                should_continue=lambda: self._material_exists(material_id),
            )

            # Visual embedding step — runs after text indexing so SigLIP does
            # not compete with BGE-M3 for RAM.  Graceful: failures are warned
            # and skipped; text pipeline result is not affected.
            if figure_items:
                await self._index_visual_figures(figure_items, material_id=str(material.id))

            await self._ensure_material_exists(material_id)
            await self._mark(
                material=material,
                job=job,
                status=PipelineStatus.INDEXED.value,
                stage=PipelineStatus.INDEXED.value,
                finished=True,
            )
            # Content changed → drop stale semantic-cache answers for this scope,
            # else a re-index keeps returning the pre-fix answer for up to the TTL.
            self._invalidate_semantic_cache(
                owner_id=str(material.owner_id),
                collection_id=str(material.collection_id) if material.collection_id else None,
            )
        except MemoryError:
            logger.critical(
                "Pipeline OOM - process may be unstable; reduce batch sizes or free RAM",
                extra={"material_id": material_id, "job_id": job_id, "stage": job.stage},
                exc_info=True,
            )
            await self._fail(
                material=material,
                job=job,
                failed_stage=job.stage,
                error="The processing pipeline failed. Please retry or inspect server logs.",
            )
            raise
        except Exception as exc:
            failed_stage = getattr(exc, "failed_stage", None) or job.stage
            logger.exception(
                "Parse/index pipeline failed",
                extra={"material_id": material_id, "job_id": job_id, "stage": failed_stage},
            )
            await self._fail(
                material=material,
                job=job,
                failed_stage=failed_stage,
                error="The processing pipeline failed. Please retry or inspect server logs.",
            )
            raise

    async def _fetch_domain_hint(self, material: Material) -> str | None:
        """Look up the configured hint field on the material's collection.

        Returns None when collection or field is missing — extractor handles
        that gracefully by skipping the domain hint block in the prompt.
        """
        field = self.settings.extraction_domain_hint_field or "subject"
        try:
            collection = await KnowledgeCollection.get(material.collection_id)
        except Exception:
            return None
        if collection is None:
            return None
        value = getattr(collection, field, None)
        return str(value).strip() if value else None

    @staticmethod
    async def _ensure_material_exists(material_id: str) -> None:
        if not await ParseIndexPipeline._material_exists(material_id):
            raise LookupError(f"Material was deleted while pipeline was running: {material_id}")

    @staticmethod
    async def _material_exists(material_id: str) -> bool:
        return await Material.get(PydanticObjectId(material_id)) is not None

    async def _contextual_enrich(
        self,
        chunks: list,
        evidence_map,
    ) -> list:
        from src.core.runtime_config import get_override
        enabled = get_override("contextual_retrieval_enabled", self.settings.contextual_retrieval_enabled)
        if not enabled or not chunks:
            return chunks
        try:
            from src.core.model_factory import build_llm
            llm = build_llm(self.settings)
            enricher = ContextualEnricher(llm, concurrency=self.settings.contextual_retrieval_concurrency)
            enriched = await enricher.enrich(chunks, evidence_map)
            enriched_count = sum(1 for c in enriched if c.contextualized_content)
            logger.info(
                "Contextual enrichment done",
                extra={"total": len(enriched), "enriched": enriched_count, "skipped": len(enriched) - enriched_count},
            )
            return enriched
        except Exception:
            logger.exception("Contextual enrichment failed entirely - continuing without context")
            return chunks

    # ── Visual embedding helpers ───────────────────────────────────────────────

    def _collect_figure_items(self, parsed: ParsedDocument, material) -> list[FigureIndexItem]:
        """Build FigureIndexItem list from figure blocks that have an image path."""
        items: list[FigureIndexItem] = []
        for block in parsed.blocks:
            if block.block_type != BlockType.FIGURE.value:
                continue
            img_path: str | None = block.extra.get("figure_image_path")
            if not img_path:
                continue
            items.append(
                FigureIndexItem(
                    owner_id=material.owner_id,
                    collection_id=str(material.collection_id),
                    material_id=str(material.id),
                    document_name=material.original_name,
                    page=block.page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    caption=block.content or "",
                    source_language=block.language,
                    bbox=block.bbox,
                    image_path=img_path,
                )
            )
        logger.info(
            "Collected figure items for visual indexing",
            extra={"count": len(items), "material_id": str(material.id)},
        )
        return items

    async def _index_visual_figures(
        self, figure_items: list[FigureIndexItem], *, material_id: str
    ) -> None:
        """Embed and index figure images with SigLIP; always unloads provider."""
        from src.rag.embedding_factory import build_visual_provider

        visual_provider = None
        try:
            visual_provider = build_visual_provider(self.settings)
            if visual_provider is None:
                return
            await self.indexer.index_visual(
                figure_items=figure_items, visual_provider=visual_provider
            )
        except Exception as exc:
            logger.warning(
                "Visual embedding failed — skipping (text index is unaffected)",
                extra={"material_id": material_id, "error": str(exc)},
            )
        finally:
            if visual_provider is not None:
                try:
                    visual_provider.unload()
                except Exception as exc:
                    logger.debug(
                        "Visual provider unload error (non-fatal)",
                        extra={"error": str(exc)},
                    )

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _resolve_chunker(self, *, file_type: str | None = None) -> LayoutAwareChunker | SemanticChunker | SlideAwareChunker | AudioChunker:
        if self._chunker_override is not None:
            return self._chunker_override
        # PPTX: use slide-boundary-preserving chunker (1 chunk per slide).
        # Semantic chunker splits across slides → fragments lose context.
        if file_type and file_type.lower() == "pptx":
            return SlideAwareChunker(self.settings)
        # Audio: group whisper segments by time window (~45s/chunk).
        # Preserves audio_start/end_seconds for citation deep-linking.
        if file_type and file_type.lower() in AUDIO_EXTENSIONS:
            return AudioChunker(self.settings)
        if self.settings.chunk_strategy == "semantic":
            embedder = None
            if self.indexer is not None and hasattr(self.indexer, "embedder"):
                embedder = self.indexer.embedder
            return SemanticChunker(self.settings, embedder=embedder)
        return LayoutAwareChunker(self.settings)

    def _free_ollama_models(self) -> None:
        """Best-effort unload of all currently-loaded Ollama models to free RAM.

        Queries /api/ps and re-requests each with keep_alive=0 (Ollama's unload
        signal). Non-fatal: a failure here just means Docling runs with less RAM.
        """
        if not self.settings.free_ollama_before_parse:
            return
        import httpx

        base = self.settings.ollama_base_url.rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                loaded = client.get(f"{base}/api/ps").json().get("models", [])
                for model in loaded:
                    name = model.get("name") or model.get("model")
                    if not name:
                        continue
                    client.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0, "prompt": ""})
            if loaded:
                import time

                logger.info("Freed %d Ollama model(s) before parse to reclaim RAM", len(loaded))
                time.sleep(3)
        except Exception as exc:
            logger.debug("Ollama unload skipped (%s)", type(exc).__name__)

    def _parse_docling(self, path: Path, *, language: str) -> ParsedDocument:
        """Parse a Docling-supported document, isolated in a fresh subprocess.

        The server process corrupts Docling's models over its lifetime (convert()
        returns an empty document → every page falls back to OCR → tables/figures/
        equations lost). A fresh subprocess parses reliably. On any subprocess
        failure (timeout, crash, bad output) we fall back to in-process parsing so
        ingestion never hard-fails.
        """
        # Free the ~4.7GB Ollama VLM first — on a RAM-tight box it starves Docling
        # → std::bad_alloc on every page → empty parse → OCR fallback loses tables.
        self._free_ollama_models()

        if not self.settings.docling_subprocess_enabled:
            return self.docling_parser.parse(path, language=language)

        import os
        import subprocess
        import sys
        import tempfile
        from uuid import uuid4

        backend_dir = Path(__file__).resolve().parents[2]  # …/backend
        out_path = Path(tempfile.gettempdir()) / f"agentbook-parse-{uuid4().hex}.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "src.processing.parse_worker", str(path), language or "unknown", str(out_path)],
                cwd=str(backend_dir),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                timeout=self.settings.docling_subprocess_timeout_seconds,
            )
            if proc.returncode == 0 and out_path.exists():
                parsed = ParsedDocument.model_validate_json(out_path.read_text(encoding="utf-8"))
                logger.info("Isolated Docling parse ok pages=%s blocks=%s", len(parsed.pages), len(parsed.blocks))
                return parsed
            logger.warning(
                "Isolated Docling parse failed (rc=%s) — falling back in-process. stderr=%s",
                proc.returncode,
                (proc.stderr or "")[-600:],
            )
        except subprocess.TimeoutExpired:
            logger.warning("Isolated Docling parse timed out — falling back in-process")
        except Exception as exc:
            logger.warning("Isolated Docling parse error (%s) — falling back in-process", type(exc).__name__)
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        return self.docling_parser.parse(path, language=language)

    def _parse_material(self, material: Material) -> ParsedDocument:
        path = self._material_path(material)
        if material.file_type in SUPPORTED_DOCLING_EXTENSIONS:
            parsed = self._parse_docling(path, language=material.language)
            from collections import Counter as _Counter
            _bt = dict(_Counter(b.block_type for b in parsed.blocks))
            # Inline in the message — the log formatter does not render `extra`.
            logger.info(
                "Parsed material %s parser=%s images_scale=%s block_types=%s",
                str(material.id),
                parsed.extra.get("parser"),
                self.settings.docling_images_scale,
                _bt,
            )
            ocr_ratio = _bt.get("ocr_text", 0) / max(len(parsed.blocks), 1)
            if ocr_ratio > 0.95 and material.file_type == "pdf":
                logger.warning(
                    "Parse result is >95%% OCR text - Docling layout model likely failed "
                    "(RAM pressure?). Tables/figures may be missing. material_id=%s",
                    str(material.id),
                )
            return parsed
        if material.file_type in SUPPORTED_SPREADSHEET_EXTENSIONS:
            return self.spreadsheet_parser.parse(path, language=material.language, display_name=material.original_name)
        if material.file_type in IMAGE_EXTENSIONS:
            source_type = str(material.extra_metadata.get("source_type", "")).lower()
            if "hand" in source_type or material.modality == "handwriting":
                return self.handwriting_reader.parse_image(path, language=self._declared_language(material.language))
            return self._parse_image_vlm_first(path, declared_language=material.language)
        if material.file_type in AUDIO_EXTENSIONS:
            return self.audio_parser.parse(path, language=self._declared_language(material.language))
        raise ValueError(f"No Phase 2 parser is configured for .{material.file_type}")

    def _caption_figures(self, parsed: ParsedDocument, language: str) -> ParsedDocument:
        """Fill in FIGURE block content using VLM captioning.

        Looks for blocks where extra["needs_captioning"] is True.
        For PDF: crops the bbox region from the pre-rendered page image.
        For image files and DOCX embedded pictures: the entire image or embedded image is captioned.
        Skips captioning silently if no vision model is available.
        Figures are captioned in parallel (up to 4 workers) to reduce wall-clock time.
        """
        import concurrent.futures

        figure_blocks = [
            b for b in parsed.blocks
            if b.block_type == BlockType.FIGURE.value and b.extra.get("needs_captioning")
        ]
        if not figure_blocks:
            return parsed

        captioner = FigureCaptioner(
            ollama_base_url=self.settings.ollama_base_url,
            language=language if language in {"vi", "en"} else "vi",
            timeout=self.settings.ollama_caption_timeout_seconds,
            image_max_side_px=self.settings.figure_image_max_side_px,
            num_ctx=self.settings.vlm_caption_num_ctx,
            list_bullet_ratio_max=self.settings.caption_list_bullet_ratio_max,
            list_bullet_max_words=self.settings.caption_list_bullet_max_words,
            ollama_model=self.settings.vlm_query_ollama_model,
        )
        # Pre-check Ollama availability once before spawning threads, so threads
        # share the cached result and don't each trigger a 5s network probe.
        captioner._detect_available_model()

        # Locate rendered page images if available (PDF pipeline puts them in cache).
        page_image_dir = self.settings.data_dir / "cache" / "pdf_page_images"

        # Persistent dir for DOCX/PPTX embedded figure images (needed by visual
        # embedding step). Only created when visual_embedding is enabled.
        figure_cache_dir: Path | None = None
        if self.settings.visual_embedding_enabled:
            figure_cache_dir = self.settings.data_dir / "cache" / "figure_images"

        def _caption_block(block):
            try:
                caption, img_path = self._caption_one_figure(
                    block, captioner, page_image_dir, parsed, figure_cache_dir
                )
                return block, caption, img_path, None
            except Exception as exc:
                return block, None, None, exc

        max_workers = min(len(figure_blocks), self.settings.figure_captioner_max_workers)
        captioned_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for block, caption, img_path, exc in pool.map(_caption_block, figure_blocks):
                if exc is not None:
                    logger.warning(
                        "Figure captioning failed for block",
                        extra={"block_id": block.block_id, "page": block.page_number, "error": str(exc)},
                    )
                else:
                    if img_path:
                        block.extra["figure_image_path"] = img_path
                    if caption:
                        block.content = caption
                        block.extra.pop("needs_captioning", None)
                        caption_source = "ocr" if caption.startswith(("[Hình", "[Figure")) else "vlm"
                        block.extra["caption_source"] = caption_source
                        block.extra["parse_method"] = "figure_caption"
                        if caption_source == "vlm" and captioner._available_model:
                            block.extra["vlm_model"] = captioner._available_model
                        if caption_source == "ocr":
                            block.extra["fallback_reason"] = "vlm_unavailable_or_empty"
                        captioned_count += 1

        if captioned_count:
            logger.info(
                "Figure captioning complete",
                extra={"captioned": captioned_count, "total_figures": len(figure_blocks), "source": parsed.source_path},
            )

        # Unload captioner's OCR engine to free RAM before visual embedding loads SigLIP.
        captioner.unload()
        return parsed

    @staticmethod
    def _caption_one_figure(
        block,
        captioner: FigureCaptioner,
        page_image_dir: Path,
        parsed: ParsedDocument,
        figure_cache_dir: Path | None = None,
    ) -> tuple[str, str | None]:
        """Return (caption, image_path | None).

        image_path is the on-disk path of the figure image so the visual
        embedding pipeline can load it later.  None when the image could not
        be persisted (PDF without page render, unsupported type, etc.).
        """
        import base64
        import tempfile
        from uuid import NAMESPACE_URL, uuid5

        source_path = Path(parsed.source_path)

        # PDF: crop the bbox region from the pre-rendered page image.
        if parsed.file_type == "pdf":
            page_img_name = f"{uuid5(NAMESPACE_URL, f'{source_path}:{block.page_number}').hex}.png"
            page_img_path = page_image_dir / page_img_name
            if page_img_path.exists():
                try:
                    import cv2
                    img = cv2.imread(str(page_img_path))
                    if img is not None:
                        ph, pw = img.shape[:2]
                        # Detect "figure-only page": Docling sometimes treats a
                        # whole scanned legal page as 1 figure block with a tiny
                        # or missing bbox. Crop-then-OCR loses the page text.
                        # When bbox is missing OR covers >70% of the page area,
                        # caption the FULL page so EasyOCR sees the whole text.
                        bbox = block.bbox
                        if bbox is None or _bbox_covers_most_of_page(bbox, pw, ph):
                            caption = captioner.caption_image_path(page_img_path)
                            return caption, str(page_img_path)
                        caption, crop_path = captioner.caption_page_region_with_path(
                            page_img_path, bbox, page_width=pw, page_height=ph
                        )
                        # If crop-based captioning came back empty (typical for
                        # scanned-text figures where the VLM/OCR couldn't read
                        # the cropped region), retry on the full page image.
                        if not (caption or "").strip():
                            caption = captioner.caption_image_path(page_img_path)
                            return caption, str(page_img_path)
                        return caption, str(crop_path) if crop_path else None
                except ImportError:
                    pass

        # Standalone image: caption the whole file; the path is already on disk.
        if parsed.file_type in IMAGE_EXTENSIONS:
            return captioner.caption_image_path(source_path), str(source_path)

        # DOCX/PPTX embedded figure: decode the data-URI.
        # When visual_embedding is enabled, save to figure_cache_dir so the file
        # persists for SigLIP.  Otherwise use a temp file (legacy behaviour).
        embedded_uri = block.extra.get("embedded_image_uri")
        if isinstance(embedded_uri, str) and embedded_uri.startswith("data:image/"):
            try:
                import hashlib as _hashlib
                header, encoded = embedded_uri.split(",", 1)
                suffix = ".png"
                if "jpeg" in header or "jpg" in header:
                    suffix = ".jpg"
                elif "webp" in header:
                    suffix = ".webp"
                data = base64.b64decode(encoded)

                if figure_cache_dir is not None:
                    figure_cache_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = _hashlib.sha1(encoded[:256].encode()).hexdigest()[:12]
                    save_path = figure_cache_dir / f"fig-{img_hash}{suffix}"
                    save_path.write_bytes(data)
                    caption = captioner.caption_image_path(save_path)
                    return caption, str(save_path)

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(data)
                    tmp_path = Path(handle.name)
                try:
                    return captioner.caption_image_path(tmp_path), None
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass

        # PDF fallback: digital PDFs have no pre-rendered page image and Docling
        # is intentionally not asked to retain picture pixels (OOM risk). Render
        # the figure's page on demand with PyMuPDF — memory-light, one page at a
        # time — so the VLM/SigLIP still get a real image to work with.
        if parsed.file_type == "pdf":
            result = ParseIndexPipeline._render_pdf_page_for_caption(
                block, source_path, captioner, figure_cache_dir
            )
            if result is not None:
                return result

        return "", None

    @staticmethod
    def _render_pdf_page_for_caption(
        block, source_path: Path, captioner: FigureCaptioner, figure_cache_dir: Path | None
    ) -> tuple[str, str | None] | None:
        """Render the figure's PDF page with PyMuPDF and caption it.

        Returns (caption, image_path) or None when rendering is unavailable.
        Full-page render (≈144 DPI) keeps the figure intact without fragile
        bbox→PDF coordinate mapping; it is plenty for VLM captioning + SigLIP.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None
        import hashlib
        import tempfile

        try:
            with fitz.open(str(source_path)) as pdf:
                page_index = max(1, getattr(block, "page_number", 1) or 1) - 1
                if not (0 <= page_index < pdf.page_count):
                    return None
                png = pdf[page_index].get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        except Exception as exc:
            logger.debug("PyMuPDF figure render failed", extra={"error": str(exc), "block": getattr(block, "block_id", None)})
            return None
        if not png or len(png) < 512:
            return None

        if figure_cache_dir is not None:
            figure_cache_dir.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha1(f"{source_path}:{getattr(block, 'block_id', '')}".encode()).hexdigest()[:12]
            save_path = figure_cache_dir / f"figpdf-{name}.png"
            save_path.write_bytes(png)
            return captioner.caption_image_path(save_path), str(save_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
            handle.write(png)
            tmp_path = Path(handle.name)
        try:
            return captioner.caption_image_path(tmp_path), None
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _parse_image_vlm_first(self, path: Path, *, declared_language: str) -> ParsedDocument:
        """For standalone image files: OCR-first for text-dense scans, VLM for visuals.

        Strategy:
          1. Run OCR first. If it yields >= OCR_TEXT_WORD_THRESHOLD words, the image
             is a printed text scan — use OCR directly (more faithful than VLM).
          2. Otherwise (diagram, infographic, chart) try VLM captioning.
          3. Fall back to OCR if VLM is unavailable or produces short/gibberish output.

        This prevents VLMs from hallucinating content on text-scan images where OCR
        is ground truth.
        """
        word_threshold = self.settings.image_ocr_text_word_threshold

        language = declared_language if declared_language in {"vi", "en"} else "vi"

        # Step 1: run OCR unconditionally — cheap and faithful for text scans.
        # OCRQualityError (low-quality image) → treat as sparse, fall through to VLM.
        ocr_parsed = None
        try:
            ocr_parsed = self._parse_printed_image(path, declared_language=declared_language)
        except OCRQualityError as exc:
            logger.info(
                "OCR quality gate failed for standalone image — trying VLM fallback",
                extra={"image": path.name, "reason": str(exc)[:120]},
            )
        ocr_word_count = sum(
            len((b.content or "").split())
            for page in (ocr_parsed.pages if ocr_parsed else [])
            for b in page.blocks
        )
        if ocr_parsed is not None and ocr_word_count >= word_threshold:
            # Text-dense. Pure text scans → OCR is faithful, skip VLM. But charts /
            # tables are ALSO text-dense, yet OCR flattens their 2D structure
            # (numbers lose their row/year association). Detect those via numeric
            # density and ADD a VLM structure pass on top of OCR (hybrid): OCR
            # keeps exact numbers, VLM reconstructs the metric↔year table.
            numeric_ratio = self._numeric_token_ratio(ocr_parsed)
            if numeric_ratio < self.settings.image_numeric_ratio_vlm_trigger:
                logger.info(
                    "Standalone image is text-dense — using OCR directly (skipping VLM)",
                    extra={"image": path.name, "ocr_words": ocr_word_count, "numeric_ratio": round(numeric_ratio, 3)},
                )
                ocr_parsed.extra.setdefault("parse_method", "ocr")
                ocr_parsed.extra["vlm_attempted"] = False
                ocr_parsed.extra["vlm_skipped_reason"] = "ocr_text_dense"
                return ocr_parsed

            logger.info(
                "Standalone image is text-dense but numeric-heavy (chart/table) — adding VLM structure pass",
                extra={"image": path.name, "ocr_words": ocr_word_count, "numeric_ratio": round(numeric_ratio, 3)},
            )
            vlm_block = self._vlm_structure_block(path, language=language)
            if vlm_block is not None and ocr_parsed.pages:
                ocr_parsed.pages[0].blocks.insert(0, vlm_block)
                ocr_parsed.extra["parse_method"] = "ocr+vlm"
                ocr_parsed.extra["vlm_attempted"] = True
                ocr_parsed.extra["vlm_structure_added"] = True
            else:
                ocr_parsed.extra.setdefault("parse_method", "ocr")
                ocr_parsed.extra["vlm_attempted"] = True
                ocr_parsed.extra["vlm_skipped_reason"] = "vlm_unavailable_or_short"
            return ocr_parsed

        # Step 2: sparse OCR → likely a visual/infographic → try VLM.
        captioner = FigureCaptioner(
            ollama_base_url=self.settings.ollama_base_url,
            language=language,
            ocr_fallback=False,
            timeout=self.settings.ollama_caption_timeout_seconds,
            image_max_side_px=self.settings.figure_image_max_side_px,
            num_ctx=self.settings.vlm_caption_num_ctx,
            list_bullet_ratio_max=self.settings.caption_list_bullet_ratio_max,
            list_bullet_max_words=self.settings.caption_list_bullet_max_words,
            ollama_model=self.settings.vlm_query_ollama_model,
        )
        vlm_model = captioner._detect_available_model()
        fallback_reason = "vlm_unavailable"
        if vlm_model:
            from src.processing.figure_captioner import _looks_like_gibberish
            caption = captioner.caption_image_path(path)
            if caption and _looks_like_gibberish(caption):
                logger.warning(
                    "VLM produced gibberish — falling back to OCR",
                    extra={"model": vlm_model, "image": path.name, "preview": caption[:120]},
                )
                fallback_reason = "vlm_gibberish_detected"
                caption = ""
            if caption and len(caption.strip()) >= 80:
                logger.info(
                    "Standalone image parsed via VLM",
                    extra={"model": vlm_model, "image": path.name, "chars": len(caption)},
                )
                block = ParsedBlock(
                    block_id=f"blk-{path.stem[:12]}",
                    block_index=0,
                    block_type=BlockType.PARAGRAPH.value,
                    content=caption,
                    page_number=1,
                    reading_order=0,
                    language=language,
                    source="vlm",
                    extra={
                        "parse_method": "vlm",
                        "caption_source": "vlm",
                        "vlm_model": vlm_model,
                        "image_file": path.name,
                    },
                )
                return ParsedDocument(
                    source_path=str(path),
                    file_type=path.suffix.lstrip(".").lower(),
                    language=language,
                    pages=[ParsedPage(page_number=1, blocks=[block])],
                    extra={"parser": "vlm", "parse_method": "vlm", "vlm_model": vlm_model},
                )
            fallback_reason = "vlm_caption_too_short_or_empty"

        # Step 3: VLM unavailable / short — return OCR result (or empty doc if OCR also failed).
        if ocr_parsed is None:
            logger.warning(
                "Both OCR quality gate and VLM failed — returning empty document",
                extra={"image": path.name, "fallback_reason": fallback_reason},
            )
            return ParsedDocument(
                source_path=str(path),
                file_type=path.suffix.lstrip(".").lower(),
                language=language,
                pages=[],
                extra={"parse_method": "none", "fallback_reason": fallback_reason},
            )
        ocr_parsed.extra.setdefault("parse_method", "ocr")
        ocr_parsed.extra["fallback_reason"] = fallback_reason
        ocr_parsed.extra["vlm_attempted"] = bool(vlm_model)
        if vlm_model:
            ocr_parsed.extra["vlm_model"] = vlm_model
        return ocr_parsed

    @staticmethod
    def _numeric_token_ratio(parsed: ParsedDocument) -> float:
        """Fraction of whitespace tokens that contain a digit.

        Charts/tables (financial pages, statistics) are number-dense; flowing
        prose is not. Used to detect text-dense-but-structured images that need a
        VLM structure pass even though OCR already returned plenty of words.
        """
        total = 0
        numeric = 0
        for page in parsed.pages:
            for block in page.blocks:
                for tok in (block.content or "").split():
                    total += 1
                    if any(ch.isdigit() for ch in tok):
                        numeric += 1
        return (numeric / total) if total else 0.0

    def _invalidate_semantic_cache(self, *, owner_id: str, collection_id: str | None) -> None:
        """Drop cached query→answer pairs for a scope after its content changes.

        Best-effort: any failure (Redis down, import error) is logged and ignored
        so it never affects the just-completed index.
        """
        try:
            from src.services.semantic_query_cache import SemanticQueryCache

            cache = SemanticQueryCache(redis_url=self.settings.redis_url)
            if cache.enabled:
                cache.invalidate_scope(owner_id=owner_id, collection_id=collection_id)
        except Exception as exc:
            logger.warning(
                "Semantic cache invalidation skipped after re-index",
                extra={"owner_id": owner_id, "collection_id": collection_id, "error": str(exc)},
            )

    def _vlm_structure_block(self, path: Path, *, language: str) -> "ParsedBlock | None":
        """Caption a chart/table image with a VLM into a structure-preserving block.

        Returns None when no vision model is available or the caption is too
        short / gibberish, so the caller keeps OCR-only output.
        """
        captioner = FigureCaptioner(
            ollama_base_url=self.settings.ollama_base_url,
            language=language,
            ocr_fallback=False,
            timeout=self.settings.ollama_caption_timeout_seconds,
            image_max_side_px=self.settings.image_structure_vlm_max_side_px,
            num_ctx=self.settings.vlm_caption_num_ctx,
            num_predict=self.settings.image_structure_vlm_num_predict,
            list_bullet_ratio_max=self.settings.caption_list_bullet_ratio_max,
            list_bullet_max_words=self.settings.caption_list_bullet_max_words,
            ollama_model=self.settings.vlm_query_ollama_model,
        )
        if not captioner._detect_available_model():
            return None
        from src.processing.figure_captioner import _looks_like_gibberish

        caption = captioner.caption_image_path(path)
        if not caption or len(caption.strip()) < 80 or _looks_like_gibberish(caption):
            return None
        # This pass only fires on numeric-dense images; a table the VLM returned
        # with (near-)empty body cells is a failure (small VLMs collapse on
        # multi-panel pages). Discard it so the caller keeps faithful OCR-only.
        if not _table_has_numeric_data(
            caption, min_digit_cell_ratio=self.settings.image_structure_vlm_min_digit_cell_ratio
        ):
            logger.warning(
                "VLM structure pass produced a dataless table — discarding (keeping OCR-only)",
                extra={"image": path.name, "preview": caption[:120]},
            )
            return None
        return ParsedBlock(
            block_id=f"blk-vlm-{path.stem[:12]}",
            block_index=0,
            block_type=BlockType.TABLE.value,
            content=caption,
            page_number=1,
            reading_order=0,
            language=language,
            source="vlm",
            extra={"parse_method": "vlm", "caption_source": "vlm_structure", "image_file": path.name},
        )

    def _vlm_table_fallback(self, parsed: ParsedDocument, language: str) -> ParsedDocument:
        """Hybrid table reader (2b): VLM-read numeric-dense figures on pages where
        Docling's TableFormer found no table.

        Docling (``table_source="docling"``) stays the exact, citation-grade
        source. This only fires for figures Docling missed; the produced table is
        tagged ``table_source="vlm"`` with a lower confidence so the
        evidence/citation layer treats its numbers cautiously. Bounded: skips
        pages that already have a Docling table, and figures below the numeric
        trigger. Failures are non-fatal.
        """
        if not self.settings.vlm_table_fallback_enabled:
            return parsed
        docling_table_pages = {
            b.page_number
            for b in parsed.blocks
            if b.block_type == BlockType.TABLE.value and (b.extra or {}).get("table_source") == "docling"
        }
        trigger = self.settings.image_numeric_ratio_vlm_trigger
        added = 0
        for page in parsed.pages:
            if page.page_number in docling_table_pages:
                continue
            for fig in list(page.blocks):
                if fig.block_type != BlockType.FIGURE.value:
                    continue
                img = (fig.extra or {}).get("figure_image_path")
                if not img or not Path(img).exists():
                    continue
                if self._text_numeric_ratio(fig.content or "") < trigger:
                    continue
                try:
                    blk = self._vlm_structure_block(Path(img), language=language)
                except Exception as exc:
                    logger.debug(
                        "VLM table fallback skipped for figure",
                        extra={"block_id": fig.block_id, "error": str(exc)},
                    )
                    blk = None
                if blk is None:
                    continue
                blk.page_number = fig.page_number
                blk.bbox = fig.bbox
                blk.reading_order = fig.reading_order
                blk.ocr_confidence = self.settings.vlm_table_confidence
                blk.extra = {**(blk.extra or {}), "table_source": "vlm", "label": "table"}
                page.blocks.append(blk)
                added += 1
        if added:
            logger.info(
                "VLM table fallback added tables (Docling missed)",
                extra={"count": added, "source": parsed.source_path},
            )
        return parsed

    @staticmethod
    def _text_numeric_ratio(text: str) -> float:
        tokens = (text or "").split()
        if not tokens:
            return 0.0
        numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
        return numeric / len(tokens)

    def _parse_printed_image(self, path: Path, *, declared_language: str) -> ParsedDocument:
        runtime_language = self._ocr_runtime_language(declared_language)
        parsed = self._ocr_engine_for_language(runtime_language).parse_image(
            path,
            language=self._declared_language(declared_language),
        )
        parsed.extra.setdefault("parse_method", "ocr")
        if declared_language != "unknown":
            return self._apply_ocr_quality_gate(parsed)

        detected = detect_document_language([block.content for block in parsed.blocks], fallback="unknown")
        if detected != "vi" or runtime_language == "vi":
            parsed.extra["ocr_language_routing"] = {"initial": runtime_language, "selected": runtime_language}
            return self._apply_ocr_quality_gate(parsed)

        vi_parsed = self._ocr_engine_for_language("vi").parse_image(path, language="vi")
        vi_parsed.extra.setdefault("parse_method", "ocr")
        selected = self._select_better_ocr_result(parsed, vi_parsed)
        selected.extra["ocr_language_routing"] = {
            "initial": runtime_language,
            "candidate": "vi",
            "selected": selected.extra.get("ocr_lang", runtime_language),
            "initial_score": self._ocr_document_quality(parsed),
            "candidate_score": self._ocr_document_quality(vi_parsed),
        }
        return self._apply_ocr_quality_gate(selected)

    def _apply_ocr_quality_gate(self, parsed: ParsedDocument) -> ParsedDocument:
        report = score_ocr_document(
            parsed,
            min_score=self.settings.min_ocr_text_quality,
            warn_score=self.settings.warn_ocr_text_quality,
        )
        parsed.extra["ocr_quality"] = {
            "score": report.score,
            "valid_char_ratio": report.valid_char_ratio,
            "meaningful_word_ratio": report.meaningful_word_ratio,
            "repetition_ratio": report.repetition_ratio,
            "symbol_density": report.symbol_density,
            "total_chars": report.total_chars,
            "warnings": report.warnings,
        }
        if not report.is_acceptable(self.settings.min_ocr_text_quality):
            raise OCRQualityError(
                f"OCR quality score {report.score:.2f} is below fail threshold "
                f"{self.settings.min_ocr_text_quality:.2f}: {report.flag_summary()}",
                score=report.score,
                threshold=self.settings.min_ocr_text_quality,
            )
        if report.warnings:
            logger.warning(
                "OCR quality gate warnings",
                extra={
                    "score": report.score,
                    "flags": report.flag_summary(),
                    "source_path": parsed.source_path,
                    "stage": "ocr_quality",
                },
            )
        return parsed

    def _ocr_engine_for_language(self, language: str) -> EasyOCREngine:
        if self.ocr_engine is not None:
            return self.ocr_engine
        if language not in self._ocr_engines:
            lang = "vi" if language == "vi" else "en"
            # For Vietnamese, attach the VietOCR recognizer when configured — it
            # reads tone marks far better than EasyOCR on scanned text. EN stays
            # on EasyOCR (VietOCR is Vietnamese-only).
            recognizer = None
            if lang == "vi" and self.settings.ocr_recognition_engine == "vietocr":
                recognizer = VietOCRRecognizer(
                    device=self.settings.ocr_vietocr_device,
                    model_name=self.settings.ocr_vietocr_model_name,
                )
            self._ocr_engines[language] = EasyOCREngine(lang=lang, gpu=self.settings.ocr_easyocr_gpu, recognizer=recognizer)
        return self._ocr_engines[language]

    @staticmethod
    def _ocr_runtime_language(language: str) -> str:
        # "vi" uses latin_g2.pth which covers all Latin scripts (vi + en).
        # Only fall back to "en" when explicitly declared English.
        return "en" if language == "en" else "vi"

    @staticmethod
    def _declared_language(language: str) -> str:
        return language if language in {"vi", "en"} else "unknown"

    @staticmethod
    def _select_better_ocr_result(first: ParsedDocument, second: ParsedDocument) -> ParsedDocument:
        return second if ParseIndexPipeline._ocr_document_quality(second) > ParseIndexPipeline._ocr_document_quality(first) else first

    @staticmethod
    def _ocr_document_quality(parsed: ParsedDocument) -> float:
        text = "\n".join(block.content for block in parsed.blocks)
        confidences = [block.ocr_confidence for block in parsed.blocks if block.ocr_confidence is not None]
        confidence_score = (sum(confidences) / len(confidences)) * 10.0 if confidences else 0.0
        detected = detect_document_language([block.content for block in parsed.blocks], fallback="unknown")
        vi_bonus = 1.5 if detected == "vi" else 0.0
        suspect_penalty = sum(char in "ēūǎåīōσ" for char in text) * 0.8
        replacement_penalty = text.count("�") * 2.0
        return confidence_score + vi_bonus - suspect_penalty - replacement_penalty

    @staticmethod
    def _apply_language_detection(parsed: ParsedDocument, *, declared_language: str) -> tuple[str, dict[str, int]]:
        should_detect = declared_language in {"unknown", "mixed", ""}
        counts: dict[str, int] = {}
        for block in parsed.blocks:
            detected = detect_block_language(block.content, fallback=block.language if block.language != "unknown" else "unknown")
            if should_detect or block.language == "unknown":
                block.language = detected
            if block.language != "unknown":
                counts[block.language] = counts.get(block.language, 0) + 1
        document_language = detect_document_language([block.content for block in parsed.blocks], fallback=declared_language or "unknown")
        parsed.language = document_language
        return document_language, counts

    def _material_path(self, material: Material) -> Path:
        storage_path = Path(material.storage_path)
        if storage_path.is_absolute():
            return storage_path
        target = self.settings.data_dir / storage_path
        if not target.exists():
            target = self.settings.data_dir.parent / storage_path
        return target

    async def _write_processed_artifacts(self, material: Material, parsed: ParsedDocument) -> None:
        output_dir = self.settings.processed_data_dir / material.owner_id / str(material.collection_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{material.id}.parsed.json"
        output_path.write_text(json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        material.extra_metadata["parsed_artifact_path"] = str(output_path.relative_to(self.settings.data_dir)).replace("\\", "/")

    @staticmethod
    async def _mark(
        *,
        material: Material,
        job: PipelineJob,
        status: str,
        stage: str,
        finished: bool = False,
    ) -> None:
        material.status = status
        material.failed_stage = None
        material.error_message = None
        material.updated_at = utc_now()
        job.status = status
        job.stage = stage
        job.last_error = None
        job.failed_stage = None
        if finished:
            job.finished_at = utc_now()
        await material.save()
        await job.save()

    @staticmethod
    async def _fail(*, material: Material, job: PipelineJob, failed_stage: str | None, error: str) -> None:
        material.status = PipelineStatus.FAILED.value
        material.failed_stage = failed_stage
        material.error_message = error
        material.retry_count += 1
        material.updated_at = utc_now()
        job.status = PipelineStatus.FAILED.value
        job.failed_stage = failed_stage
        job.last_error = error
        job.retry_count += 1
        job.finished_at = utc_now()
        await material.save()
        await job.save()


