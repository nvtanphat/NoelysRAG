from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_dotenv_into_environ() -> None:
    env_path = project_root() / "backend" / ".env"
    if not env_path.exists():
        env_path = project_root() / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val


_load_dotenv_into_environ()


def load_yaml_config(name: str) -> dict[str, Any]:
    config_path = project_root() / "config" / name
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")
    return data


def env_value(name: str, fallback: Any) -> Any:
    return os.getenv(f"AGENTBOOK_{name}", fallback)


def env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(f"AGENTBOOK_{name}")
    if raw is None:
        return fallback
    return raw.strip().lower() not in ("false", "0", "no", "")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENTBOOK_",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    app_name: str = "Noelys"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    testing: bool = False
    api_auth_enabled: bool = False
    api_key: str | None = None
    seed_user_email: str | None = None
    seed_user_password: str | None = None

    mongodb_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mongodb_uri", "MONGODB_URI", "AGENTBOOK_MONGODB_URI"),
    )
    mongodb_database: str = "agentbook"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "agentbook_chunks"
    qdrant_timeout_seconds: int = 60

    redis_url: str = "redis://localhost:6379/0"

    data_dir: Path = Field(default_factory=lambda: project_root() / "data")
    raw_data_dir: Path = Field(default_factory=lambda: project_root() / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: project_root() / "data" / "processed")

    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "pptx", "png", "jpg", "jpeg", "csv", "xlsx", "xls"]
    )
    max_upload_size_mb: int = 20
    upload_dedupe_scope: str = "owner"
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ])

    parse_version: str = "docling-2026-04"
    chunk_version: str = "semantic_v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dense_size: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024
    embedding_use_fp16: bool = False
    normalize_embeddings: bool = True
    embedding_version: str = "bge-m3-v1"
    index_version: str = "qdrant_dense_sparse_v1"
    index_batch_size: int = 64
    llm_default_provider: str = "local"
    llm_local_model: str = "qwen2.5:3b"
    llm_extraction_provider: str = ""   # "" = same as llm_default_provider
    llm_extraction_local_model: str = ""  # "" = same as llm_local_model
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1024
    llm_timeout_seconds: float = 180.0
    reranker_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_max_pairs: int = 100
    reranker_max_length: int = 512  # truncate each (query, chunk) pair so long chunks (e.g. big tables) don't blow a predict batch up to minutes
    dense_top_k: int = 20
    sparse_top_k: int = 20
    graph_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    rerank_input_k: int = 20
    max_chunks_per_doc: int = 3
    graph_max_hops: int = 2
    agentic_rag_enabled: bool = False
    agentic_planner_llm_enabled: bool = False
    agentic_max_retrieval_iterations: int = 2
    agentic_anaphora_resolution_enabled: bool = True
    # Coordinator-only — confidence threshold below which the legacy critic
    # agent fires (after guardrails). Set in retrieval_config.yaml under
    # `retrieval.agentic_critic_activation_confidence`.
    agentic_critic_activation_confidence: float = 0.65
    # When false, skip the post-synthesis CriticAgent loop entirely. Saves
    # 30-90s per query at the cost of one self-review pass.
    agentic_critic_enabled: bool = True
    multi_query_enabled: bool = False
    smart_reranker_enabled: bool = False
    smart_reranker_threshold: float = 0.7
    # HyDE: for VI queries, generate a hypothetical English passage and use it as
    # an extra retrieval signal (retrieval-only) to bridge VI→EN. Adds one LLM
    # call per VI query (cached); set false to disable if latency-sensitive.
    hyde_enabled: bool = True
    # When true, a VI query that the static translation dict can't handle is
    # translated to English by the LLM so cross-lingual retrieval (VI query over
    # EN sources) works for arbitrary documents, not just known ML vocabulary.
    cross_lingual_llm_translation_enabled: bool = True
    llm_router_enabled: bool = False
    crag_evaluator_enabled: bool = False
    crag_correct_threshold: float = 0.55
    crag_incorrect_threshold: float = 0.25
    crag_llm_enabled: bool = False
    # Budget-aware retrieval: skip optional sub-questions when phase-1 is strong
    budget_phase2_enabled: bool = True
    budget_min_strong_score: float = 0.045
    budget_min_strong_count: int = 3
    # Semantic dedup threshold for near-duplicate content removal (Jaccard)
    retrieval_semantic_dedup_threshold: float = 0.85
    # Self-consistency: generate N candidates and vote when confidence < threshold
    agentic_self_consistency_n: int = 1       # 1 = disabled (single-shot)
    agentic_self_consistency_threshold: float = 0.65
    self_rag_reflection_enabled: bool = False
    min_reranker_score: float = 0.35
    min_evidence_confidence: float = 0.55
    min_graph_confidence: float = 0.55
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_always_eager: bool = False
    chunk_strategy: str = "semantic"
    contextual_retrieval_enabled: bool = True
    contextual_retrieval_concurrency: int = 4
    chunk_target_token_count: int = 512
    chunk_min_token_count: int = 100
    chunk_overlap_token_count: int = 50
    chunk_max_blocks_per_chunk: int = 20
    semantic_chunk_breakpoint_percentile: float = 95.0
    min_ocr_text_quality: float = 0.35
    warn_ocr_text_quality: float = 0.55
    min_handwriting_quality_score: float = 0.72
    min_handwriting_confidence: float = 0.8
    min_blur_variance: float = 80.0
    min_brightness: float = 45.0
    max_brightness: float = 230.0
    min_contrast: float = 18.0
    max_abs_skew_degrees: float = 12.0

    # Structure-adaptive visualization config (see config/viz_config.yaml).
    # Loaded as a nested dict (ladders/patterns/thresholds) — consumed by the
    # structure detector. Empty dict = detector uses built-in defaults.
    viz_config: dict = Field(default_factory=dict)

    # Entity extraction config (see config/extraction_config.yaml)
    extraction_mode: str = "dynamic"  # simple | dynamic | strict
    extraction_default_entity_types: list[str] = Field(
        default_factory=lambda: [
            "concept", "person", "organization", "location",
            "event", "artifact", "time", "quantity",
        ]
    )
    extraction_output_format: str = "json"  # json | delimiter
    extraction_max_gleanings: int = 0
    extraction_domain_hint_field: str = "subject"
    extraction_few_shots: list[dict] = Field(default_factory=list)
    extraction_entity_backend: str = "llm"  # "llm" | "gliner"
    extraction_gliner_model: str = "gliner-community/gliner_medium-v2.5"
    extraction_gliner_threshold: float = 0.5
    extraction_ontology: dict = Field(default_factory=dict)  # entity_type → valid relation types
    extraction_graph_entity_types: list[str] = Field(default_factory=list)  # types that enter the graph
    extraction_relation_types: list[str] = Field(default_factory=list)      # relation vocab
    extraction_entity_type_descriptions: dict = Field(default_factory=dict)  # type → 1-line desc for refiner prompt
    extraction_entity_refine_enabled: bool = True            # ontology-guided LLM entity refinement pass
    extraction_entity_refine_max_entities: int = 200         # cap entities sent to the refiner LLM call
    extraction_excluded_sections: list[str] = Field(default_factory=list)   # heading patterns skipped for KG
    extraction_merge_threshold: float = 0.82                # cross-type BGE-M3 merge
    extraction_cross_lingual_merge_threshold: float = 0.74  # same-type BGE-M3 merge

    # Knowledge-graph endpoint thresholds + fetch caps (see retrieval_config.yaml → graph)
    graph_fuzzy_dedup_threshold: float = 0.88
    graph_semantic_dedup_threshold: float = 0.82
    graph_display_min_confidence: float = 0.5
    graph_display_min_mentions: int = 2
    graph_max_entities_fetch: int = 400
    graph_max_relations_fetch: int = 400
    graph_max_visible_nodes: int = 50
    graph_focus_primary_cap: int = 15
    graph_focus_neighbor_cap: int = 12
    graph_focus_fallback_cap: int = 20
    graph_mindmap_entity_fetch: int = 160
    graph_mindmap_entity_cap: int = 60
    graph_semantic_relation_max_concepts: int = 25
    graph_semantic_relation_max_passages: int = 18
    graph_semantic_relation_max_passage_chars: int = 600
    graph_semantic_relation_gleaning: bool = False
    graph_min_knowledge_edges_for_no_fallback: int = 3

    # Phase B — Adaptive Retrieval Budget (skip sparse + graph on easy factuals)
    adaptive_retrieval_enabled: bool = True
    adaptive_dense_skip_threshold: float = 0.55
    adaptive_strong_hits_required: int = 3
    adaptive_strong_hit_min_score: float = 0.35
    adaptive_eligible_routes: list[str] = Field(
        default_factory=lambda: ["factual", "general", "claim_check"]
    )

    # Modality-aware routing (config/retrieval_config.yaml → routing.modality)
    modality_routing_enabled: bool = True
    modality_table_keywords: list[str] = Field(default_factory=list)
    modality_figure_keywords: list[str] = Field(default_factory=list)
    modality_audio_keywords: list[str] = Field(default_factory=list)
    modality_table_boost_multiplier: float = 0.30
    modality_extra_pass: bool = True
    modality_filtered_pass_limit_ratio: float = 0.5
    vlm_list_caption_score_penalty: float = 0.40
    router_agentic_confidence_threshold: float = 0.55

    # Sentence-level Evidence Coverage (SLEC) — Adaptive Evidence-Guided RAG core
    slec_enabled: bool = True
    slec_supported_threshold: float = 0.55
    slec_partial_threshold: float = 0.30
    slec_refuse_below: float = 0.40
    slec_drop_unsupported: bool = True
    slec_min_sentence_chars: int = 12
    slec_max_sentences: int = 24
    slec_skip_routes: list[str] = Field(default_factory=lambda: ["claim_check"])

    # Quality gate thresholds (formerly hardcoded in quality_gate.py)
    quality_gate_conf_caution: float = 0.50
    quality_gate_conf_fail: float = 0.30
    quality_gate_slec_caution: float = 0.60
    quality_gate_slec_fail: float = 0.40
    quality_gate_citation_caution: float = 0.80
    quality_gate_citation_fail: float = 0.50

    # Visual caption fallback: when SigLIP visual index has no hits, fall back to
    # text retrieval for figure queries rather than refusing immediately.
    visual_caption_fallback_enabled: bool = True

    # Post-retrieval graph probe — adaptive graph augmentation for all routes
    graph_probe_enabled: bool = True
    graph_probe_min_entities: int = 2
    graph_probe_top_chunks_to_scan: int = 5
    graph_probe_max_graph_chunks: int = 4
    graph_probe_skip_routes: list[str] = Field(default_factory=lambda: ["claim_check"])

    # Visual embedding (SigLIP)
    visual_embedding_enabled: bool = False
    visual_embedding_model: str = "google/siglip-base-patch16-224"
    visual_embedding_device: str = "cpu"
    visual_embedding_dense_size: int = 768
    visual_embedding_batch_size: int = 4
    visual_embedding_backend: str = "pytorch"
    qdrant_visual_collection_name: str = "agentbook_visual"
    visual_retrieval_top_k: int = 3
    lexical_fallback_multiplier: int = 3
    chunk_min_content_chars: int = 50
    figure_captioner_max_workers: int = 4
    ollama_caption_timeout_seconds: float = 300.0
    figure_image_max_side_px: int = 1024
    # Ollama context window for VLM captioning. A 2048px structure-pass image
    # produces ~3.7k vision tokens + prompt → overflows Ollama's 4096 default
    # and returns a 400 exceed_context_size_error. Must comfortably exceed
    # (vision tokens at image_structure_vlm_max_side_px + prompt + num_predict).
    vlm_caption_num_ctx: int = 8192
    caption_list_bullet_ratio_max: float = 0.70
    caption_list_bullet_max_words: int = 3
    memory_max_turns: int = 10

    ocr_text_detection_model_name: str = "PP-OCRv5_mobile_det"
    ocr_text_recognition_model_name: str | None = None
    ocr_text_det_limit_side_len: int = 1280
    ocr_text_det_limit_type: str = "max"
    ocr_rec_score_threshold: float = 0.5
    ocr_enable_grayscale_variant: str = "auto"
    ocr_grayscale_trigger_confidence: float = 0.85
    ocr_recognition_engine: str = "easyocr"  # easyocr | vietocr (vi-only)
    ocr_vietocr_model_name: str = "vgg_transformer"
    ocr_vietocr_device: str = "cpu"
    ocr_easyocr_gpu: bool = False
    ocr_garbled_vi_diacritic_ratio: float = 0.03  # re-OCR page when VI diacritics < 3% of alpha chars
    image_ocr_text_word_threshold: int = 20        # standalone image with >= this many OCR words = text scan
    image_numeric_ratio_vlm_trigger: float = 0.30  # text-dense image with >= this numeric-token ratio = chart/table → also run VLM
    image_structure_vlm_max_side_px: int = 2048    # resolution for the chart/table VLM structure pass (dense numbers need detail)
    image_structure_vlm_min_digit_cell_ratio: float = 0.15  # a body column is "numeric" if >= this fraction of its cells hold a digit; a VLM structure table needs >= 2 numeric columns, else discarded as a hollow grid
    image_structure_vlm_num_predict: int = 1536  # max output tokens for the chart/table structure pass; dense multi-metric pages need more than the default 512 or the table truncates
    # Column/panel-aware OCR layout: split multi-panel pages (dashboards, side-by-side
    # charts) into columns by vertical-whitespace gutters so adjacent panels are not
    # merged across columns. All thresholds are ratios of page geometry (no hardcoded px).
    layout_column_detection_enabled: bool = True
    layout_column_min_gutter_ratio: float = 0.04   # min gutter width as fraction of content width to count as a column separator
    layout_column_min_band_occupancy_ratio: float = 0.10  # min vertical occupancy for an x-bin to count as "filled" (gutter ~0; column >0)
    # Hybrid table reader (2b): VLM-read numeric-dense figures on pages where
    # Docling's TableFormer found no table. Docling stays the exact source; VLM
    # output is tagged table_source="vlm" + lower confidence so evidence/citation
    # treats its numbers with caution.
    vlm_table_fallback_enabled: bool = True
    vlm_table_confidence: float = 0.5
    pdf_render_scale: float = 1.5

    # Audio (faster-whisper)
    audio_whisper_model: str = "small"
    audio_whisper_device: str = "cpu"
    audio_whisper_compute_type: str = "int8"
    audio_whisper_beam_size: int = 5
    audio_whisper_vad_filter: bool = True
    audio_chunk_target_seconds: float = 45.0
    audio_chunk_min_seconds: float = 10.0

    # Inference engine tuning (see config/model_config.yaml → inference:)
    inference_default_answer_language: str = "vi"
    inference_default_chitchat_answer: str = "Xin chào! Tôi có thể giúp gì cho bạn?"
    inference_graph_timeout_seconds: float = 25.0
    inference_visual_inline_limit: int = 4
    inference_visual_inline_max: int = 2
    inference_retry_evidence_snippet_chars: int = 300
    inference_fallback_snippet_chars: int = 200
    inference_retry_evidence_limit: int = 5
    inference_graph_priority_score_boost: float = 0.25
    inference_graph_boost_score: float = 0.10
    inference_min_chunk_chars: int = 40
    inference_multi_doc_min_sources: int = 3
    inference_synthesis_route_types: list[str] = Field(
        default_factory=lambda: ["summarization", "comparison", "graph_relation"]
    )
    inference_self_rag_evidence_char_limit: int = 3000
    inference_self_rag_answer_char_limit: int = 2000
    inference_memory_context_header: str = "LỊCH SỬ LIÊN QUAN"
    inference_language_names: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "tiếng Việt", "en": "English", "zh": "Chinese",
            "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
        }
    )
    inference_route_prompt_map: dict[str, str] = Field(
        default_factory=lambda: {
            "summarization": "summarization.txt",
            "comparison": "comparison.txt",
            "claim_check": "claim_check.txt",
            "graph_relation": "graph_relation.txt",
        }
    )
    inference_multi_source_prompt_file: str = "multi_source.txt"
    inference_default_prompt_file: str = "qa_grounded.txt"
    inference_substantive_chunk_filter_prefixes: list[str] = Field(
        default_factory=lambda: [
            "Ghi nhớ", "Note:", "Starter Pack", "Tập trung", "Người mới",
            "nên bắt đầu", "scikit-learn User Guide", "Dive into Deep",
            "Xem thêm tại", "Link:", "URL:",
        ]
    )

    # Docling PDF converter (see config/model_config.yaml → docling:)
    vlm_query_enabled: bool = True
    vlm_query_provider: str = "auto"
    vlm_query_ollama_model: str = ""
    vlm_query_openai_model: str = ""
    vlm_query_max_images: int = 2
    vlm_query_image_max_side_px: int = 1024
    vlm_query_timeout_seconds: float = 300.0
    vlm_query_caption_fallback: bool = True
    vlm_query_verify_enabled: bool = True
    vlm_query_verify_refuse_confidence: float = 0.75

    docling_queue_max_size: int = 2
    docling_images_scale: float = 1.0
    # Run Docling parse in an isolated subprocess. A long-running server process
    # corrupts Docling's models over time (convert() returns an empty document →
    # every page falls back to OCR → all table/figure/equation structure lost). A
    # fresh subprocess parses reliably. Falls back to in-process on any failure.
    docling_subprocess_enabled: bool = True
    docling_subprocess_timeout_seconds: float = 900.0
    # Unload Ollama models (the ~4.7GB VLM) before the memory-heavy Docling parse.
    # On a RAM-tight box the loaded VLM starves Docling → std::bad_alloc on every
    # page → empty parse → OCR fallback wipes all table/figure/equation structure.
    free_ollama_before_parse: bool = True

    # Structural KG linker — connects content blocks to their enclosing section
    # heading via `belongs_to` edges (pure positional, no ML).
    structure_linker_enabled: bool = True
    structure_section_confidence: float = 0.9
    structure_belongs_to_confidence: float = 0.85

    # User-facing messages (language-keyed, see config/guardrails_config.yaml → messages:)
    messages_refusal_answer: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "Tôi không tìm thấy đủ bằng chứng trong tài liệu được cung cấp để trả lời câu hỏi này.",
            "en": "I could not find sufficient evidence in the provided documents to answer this question.",
        }
    )
    messages_refusal_prefix: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "Tôi không tìm thấy đủ bằng chứng",
            "en": "I could not find sufficient evidence",
        }
    )
    messages_low_confidence_warning: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "\n\n> Cảnh báo: Một số nhận định chưa được bằng chứng hỗ trợ trực tiếp.",
            "en": "\n\n> Warning: Some claims may not be directly supported by the evidence.",
        }
    )
    messages_partial_confidence_warning: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "\n\n> ⚠️ Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra lại nguồn gốc.",
            "en": "\n\n> ⚠️ This answer is based on limited-confidence evidence. Please verify the sources.",
        }
    )
    messages_insufficient_evidence_refusal: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "Không đủ bằng chứng đáng tin cậy để tạo câu trả lời có citation hợp lệ.",
            "en": "Insufficient reliable evidence to produce a validly-cited answer.",
        }
    )
    messages_self_rag_unsupported_prefix: dict[str, str] = Field(
        default_factory=lambda: {
            "vi": "⚠️ Chưa có đủ bằng chứng",
            "en": "⚠️ Insufficient evidence",
        }
    )

    # Agentic critic loop (see config/retrieval_config.yaml → retrieval:)
    agentic_critic_max_follow_ups: int = 2
    agentic_critic_context_buffer: int = 4

    # Extraction heuristics (see config/extraction_config.yaml → heuristics:)
    extraction_max_chars_per_llm_batch: int = 3000
    extraction_max_entity_words: int = 7
    extraction_stopword_only_max_words: int = 2
    extraction_min_confidence: float = 0.5
    extraction_merge_confidence_boost: float = 0.02
    extraction_merge_confidence_max: float = 0.97
    extraction_confidence_method_keyword: float = 0.78
    extraction_confidence_metric: float = 0.72
    extraction_confidence_capitalized_term: float = 0.55
    extraction_confidence_vietnamese_ner: float = 0.68
    extraction_method_keywords: list[str] = Field(default_factory=list)
    extraction_metric_terms: list[str] = Field(default_factory=list)
    extraction_en_stopwords: list[str] = Field(default_factory=list)
    extraction_vi_stopwords: list[str] = Field(default_factory=list)
    extraction_compound_heads: list[str] = Field(default_factory=list)
    extraction_anaphora_pattern: str = ""
    extraction_structural_junk_patterns: list[str] = Field(default_factory=list)

    # OCR preprocessing (see config/model_config.yaml → ocr.preprocessing:)
    ocr_preprocessing_min_long_edge_px: int = 1600
    ocr_preprocessing_low_contrast_threshold: float = 60.0
    ocr_preprocessing_denoise_h: int = 8
    ocr_preprocessing_denoise_template_window: int = 7
    ocr_preprocessing_denoise_search_window: int = 21
    ocr_preprocessing_clahe_clip_limit: float = 2.5
    ocr_preprocessing_clahe_tile_grid: list[int] = Field(default_factory=lambda: [8, 8])
    ocr_preprocessing_unsharp_sigma: float = 2.0
    ocr_preprocessing_unsharp_alpha: float = 1.4
    ocr_preprocessing_unsharp_beta: float = -0.4
    ocr_preprocessing_adaptive_block_size: int = 25
    ocr_preprocessing_adaptive_c: int = 10
    ocr_preprocessing_line_tolerance_px: int = 15
    ocr_preprocessing_dedup_jaccard_threshold: float = 0.80
    ocr_preprocessing_dedup_subset_coverage_threshold: float = 0.85
    ocr_preprocessing_dedup_confidence_boost_margin: float = 0.05
    ocr_easyocr_language_map: dict[str, list[str]] = Field(
        default_factory=lambda: {"vi": ["vi"], "en": ["en"]}
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @model_validator(mode="after")
    def validate_chunking_config(self) -> "Settings":
        """Validate chunking configuration constraints."""
        if self.app_env.lower() == "production":
            if not self.api_auth_enabled:
                raise ValueError("api_auth_enabled must be true when app_env is production")
            if not self.api_key:
                raise ValueError("api_key must be configured when app_env is production")
        for name, value, upper in [
            ("dense_top_k", self.dense_top_k, 100),
            ("sparse_top_k", self.sparse_top_k, 100),
            ("graph_top_k", self.graph_top_k, 50),
            ("final_top_k", self.final_top_k, 20),
            ("rerank_input_k", self.rerank_input_k, 100),
            ("reranker_max_pairs", self.reranker_max_pairs, 200),
        ]:
            if not 1 <= int(value) <= upper:
                raise ValueError(f"{name} ({value}) must be between 1 and {upper}")
        self.upload_dedupe_scope = (self.upload_dedupe_scope or "owner").strip().lower()
        if self.upload_dedupe_scope not in {"owner", "collection"}:
            raise ValueError("upload_dedupe_scope must be either 'owner' or 'collection'")
        if self.llm_max_output_tokens < 1 or self.llm_max_output_tokens > 8192:
            raise ValueError("llm_max_output_tokens must be between 1 and 8192")
        if self.chunk_min_token_count >= self.chunk_target_token_count:
            raise ValueError(
                f"chunk_min_token_count ({self.chunk_min_token_count}) must be less than "
                f"chunk_target_token_count ({self.chunk_target_token_count})"
            )
        if self.chunk_target_token_count > self.embedding_max_length:
            raise ValueError(
                f"chunk_target_token_count ({self.chunk_target_token_count}) must not exceed "
                f"embedding_max_length ({self.embedding_max_length})"
            )
        if self.chunk_overlap_token_count >= self.chunk_min_token_count:
            raise ValueError(
                f"chunk_overlap_token_count ({self.chunk_overlap_token_count}) must be less than "
                f"chunk_min_token_count ({self.chunk_min_token_count})"
            )
        if self.chunk_overlap_token_count < 0:
            raise ValueError(f"chunk_overlap_token_count ({self.chunk_overlap_token_count}) must be non-negative")
        if self.chunk_min_token_count < 1:
            raise ValueError(f"chunk_min_token_count ({self.chunk_min_token_count}) must be at least 1")
        if self.chunk_max_blocks_per_chunk < 1:
            raise ValueError(f"chunk_max_blocks_per_chunk ({self.chunk_max_blocks_per_chunk}) must be at least 1")
        if not 0 < self.semantic_chunk_breakpoint_percentile <= 100:
            raise ValueError(
                f"semantic_chunk_breakpoint_percentile ({self.semantic_chunk_breakpoint_percentile}) "
                f"must be between 0 and 100"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_config = load_yaml_config("model_config.yaml")
    retrieval_config = load_yaml_config("retrieval_config.yaml")
    guardrails_config = load_yaml_config("guardrails_config.yaml")

    upload_config = guardrails_config.get("upload", {})
    refusal_config = guardrails_config.get("refusal", {})
    image_quality_config = guardrails_config.get("image_quality", {})
    slec_config = guardrails_config.get("sentence_coverage", {})
    quality_gate_config = guardrails_config.get("quality_gate", {})
    visual_guardrails_config = guardrails_config.get("visual", {})
    graph_probe_config = retrieval_config.get("graph_probe", {})
    embedding_config = model_config.get("embedding", {})
    llm_config = model_config.get("llm", {})
    reranker_config = model_config.get("reranker", {})
    versions = model_config.get("versions", {})
    ocr_config = model_config.get("ocr", {})
    pdf_config = model_config.get("pdf", {})
    figure_captioner_config = model_config.get("figure_captioner", {})
    memory_config = model_config.get("memory", {})
    visual_config = model_config.get("visual_embedding", {})
    audio_config = model_config.get("audio", {})
    qdrant_config = retrieval_config.get("qdrant", {})
    retrieval_section = retrieval_config.get("retrieval", {})
    chunking_config = retrieval_config.get("chunking", {})
    adaptive_retrieval_config = retrieval_config.get("adaptive_retrieval", {})
    crag_config = retrieval_config.get("crag", {})
    graph_config = retrieval_config.get("graph", {})
    graph_semrel_config = graph_config.get("semantic_relation", {})
    modality_config = retrieval_config.get("routing", {}).get("modality", {})
    try:
        _ext_yaml = load_yaml_config("extraction_config.yaml")
    except FileNotFoundError:
        _ext_yaml = {}
    extraction_config = _ext_yaml.get("extraction", {})
    structure_linker_cfg = _ext_yaml.get("structure_linker", {})
    heuristics_cfg = _ext_yaml.get("heuristics", {})
    method_keywords_cfg = list(_ext_yaml.get("method_keywords", []))
    metric_terms_cfg = list(_ext_yaml.get("metric_terms", []))
    en_stopwords_cfg = list(_ext_yaml.get("en_stopwords", []))
    vi_stopwords_cfg = list(_ext_yaml.get("vi_stopwords", []))
    compound_heads_cfg = list(_ext_yaml.get("compound_heads", []))
    anaphora_pattern_cfg = str(_ext_yaml.get("anaphora_pattern", "") or "")
    structural_junk_patterns_cfg = list(_ext_yaml.get("structural_junk_patterns", []))
    try:
        viz_config = load_yaml_config("viz_config.yaml").get("visualization", {})
    except FileNotFoundError:
        viz_config = {}
    inference_cfg = model_config.get("inference", {})
    messages_config = guardrails_config.get("messages", {})
    docling_cfg = model_config.get("docling", {})
    ocr_preproc_cfg = ocr_config.get("preprocessing", {})

    return Settings(
        max_upload_size_mb=upload_config.get("max_file_size_mb", 20),
        upload_dedupe_scope=str(upload_config.get("dedupe_scope", "owner")),
        allowed_upload_extensions=upload_config.get(
            "allowed_extensions", ["pdf", "docx", "pptx", "png", "jpg", "jpeg", "csv", "xlsx", "xls"]
        ),
        embedding_model=embedding_config.get("model_name", "BAAI/bge-m3"),
        embedding_dense_size=embedding_config.get("dense_size", 1024),
        embedding_device=env_value("EMBEDDING_DEVICE", embedding_config.get("device", "cpu")),
        embedding_batch_size=int(env_value("EMBEDDING_BATCH_SIZE", embedding_config.get("batch_size", 8))),
        embedding_max_length=embedding_config.get("max_length", 1024),
        embedding_use_fp16=env_bool("EMBEDDING_USE_FP16", embedding_config.get("use_fp16", False)),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        embedding_version=embedding_config.get("embedding_version", "bge-m3-v1"),
        qdrant_collection_name=qdrant_config.get("collection_name", "agentbook_chunks"),
        parse_version=versions.get("parse_version", "docling-2026-04"),
        chunk_version=versions.get("chunk_version", "semantic_v1"),
        index_version=versions.get("index_version", "qdrant_dense_sparse_v1"),
        index_batch_size=qdrant_config.get("index_batch_size", 64),
        llm_default_provider=env_value("LLM_DEFAULT_PROVIDER", llm_config.get("default_provider", "local")),
        llm_local_model=env_value("LLM_LOCAL_MODEL", llm_config.get("local_model", "qwen2.5:3b")),
        llm_extraction_provider=str(llm_config.get("extraction_provider", "")),
        llm_extraction_local_model=str(llm_config.get("extraction_local_model", "")),
        ollama_base_url=env_value("OLLAMA_BASE_URL", llm_config.get("ollama_base_url", "http://localhost:11434")),
        openai_base_url=env_value("OPENAI_BASE_URL", llm_config.get("openai_base_url", "https://api.openai.com/v1")),
        openai_model=env_value("OPENAI_MODEL", llm_config.get("openai_model", "gpt-4o-mini")),
        llm_temperature=llm_config.get("temperature", 0.1),
        llm_max_output_tokens=llm_config.get("max_output_tokens", 1024),
        llm_timeout_seconds=float(env_value("LLM_TIMEOUT_SECONDS", llm_config.get("timeout_seconds", 180.0))),
        reranker_enabled=env_bool("RERANKER_ENABLED", reranker_config.get("enabled", True)),
        reranker_model_name=reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3"),
        reranker_device=env_value("RERANKER_DEVICE", reranker_config.get("device", "cpu")),
        reranker_max_pairs=int(reranker_config.get("max_pairs", 80)),
        reranker_max_length=int(reranker_config.get("max_length", 512)),
        dense_top_k=retrieval_section.get("dense_top_k", 20),
        sparse_top_k=retrieval_section.get("sparse_top_k", 20),
        graph_top_k=retrieval_section.get("graph_top_k", 10),
        final_top_k=retrieval_section.get("final_top_k", retrieval_section.get("top_k", 5)),
        rrf_k=retrieval_section.get("rrf_k", 60),
        rerank_input_k=retrieval_section.get("rerank_input_k", 15),
        graph_max_hops=min(int(retrieval_section.get("graph_max_hops", 2)), 2),
        agentic_rag_enabled=env_bool("AGENTIC_RAG_ENABLED", retrieval_section.get("agentic_rag_enabled", False)),
        agentic_planner_llm_enabled=env_bool("AGENTIC_PLANNER_LLM_ENABLED", retrieval_section.get("agentic_planner_llm_enabled", False)),
        agentic_max_retrieval_iterations=int(retrieval_section.get("agentic_max_retrieval_iterations", 2)),
        agentic_anaphora_resolution_enabled=bool(retrieval_section.get("agentic_anaphora_resolution_enabled", True)),
        agentic_critic_activation_confidence=float(retrieval_section.get("agentic_critic_activation_confidence", 0.65)),
        agentic_critic_enabled=env_bool("AGENTIC_CRITIC_ENABLED", retrieval_section.get("agentic_critic_enabled", True)),
        crag_evaluator_enabled=env_bool("CRAG_EVALUATOR_ENABLED", crag_config.get("evaluator_enabled", False)),
        crag_correct_threshold=float(crag_config.get("correct_threshold", 0.55)),
        crag_incorrect_threshold=float(crag_config.get("incorrect_threshold", 0.25)),
        crag_llm_enabled=env_bool("CRAG_LLM_ENABLED", crag_config.get("llm_enabled", False)),
        budget_phase2_enabled=env_bool(
            "BUDGET_PHASE2_ENABLED",
            retrieval_config.get("budget_aware_retrieval", {}).get("phase2_enabled", True),
        ),
        budget_min_strong_score=float(
            retrieval_config.get("budget_aware_retrieval", {}).get("min_strong_score", 0.045)
        ),
        budget_min_strong_count=int(
            retrieval_config.get("budget_aware_retrieval", {}).get("min_strong_count", 3)
        ),
        retrieval_semantic_dedup_threshold=float(
            retrieval_config.get("semantic_dedup", {}).get("threshold", 0.85)
        ),
        multi_query_enabled=env_bool("MULTI_QUERY_ENABLED", retrieval_section.get("multi_query_enabled", False)),
        api_auth_enabled=env_bool("API_AUTH_ENABLED", str(env_value("APP_ENV", "development")).lower() == "production"),
        api_key=env_value("API_KEY", None),
        seed_user_email=env_value("SEED_USER_EMAIL", None),
        seed_user_password=env_value("SEED_USER_PASSWORD", None),
        min_ocr_text_quality=refusal_config.get("min_ocr_text_quality", 0.35),
        warn_ocr_text_quality=refusal_config.get("warn_ocr_text_quality", 0.55),
        min_reranker_score=refusal_config.get("min_reranker_score", 0.35),
        min_evidence_confidence=refusal_config.get("min_evidence_confidence", 0.55),
        min_graph_confidence=refusal_config.get("min_graph_confidence", 0.55),
        chunk_strategy=chunking_config.get("strategy", "semantic"),
        contextual_retrieval_enabled=env_bool("CONTEXTUAL_RETRIEVAL_ENABLED", chunking_config.get("contextual_retrieval_enabled", True)),
        contextual_retrieval_concurrency=int(chunking_config.get("contextual_retrieval_concurrency", 4)),
        chunk_target_token_count=chunking_config.get("target_token_count", 512),
        chunk_min_token_count=chunking_config.get("min_token_count", 100),
        chunk_overlap_token_count=chunking_config.get("overlap_token_count", 50),
        chunk_max_blocks_per_chunk=chunking_config.get("max_blocks_per_chunk", 20),
        semantic_chunk_breakpoint_percentile=float(chunking_config.get("breakpoint_percentile", 95.0)),
        adaptive_retrieval_enabled=env_bool(
            "ADAPTIVE_RETRIEVAL_ENABLED",
            adaptive_retrieval_config.get("enabled", True),
        ),
        adaptive_dense_skip_threshold=float(
            adaptive_retrieval_config.get("dense_skip_threshold", 0.55)
        ),
        adaptive_strong_hits_required=int(
            adaptive_retrieval_config.get("strong_hits_required", 3)
        ),
        adaptive_strong_hit_min_score=float(
            adaptive_retrieval_config.get("strong_hit_min_score", 0.35)
        ),
        adaptive_eligible_routes=list(
            adaptive_retrieval_config.get(
                "eligible_routes", ["factual", "general", "claim_check"]
            )
        ),
        modality_routing_enabled=bool(modality_config.get("enabled", True)),
        modality_table_keywords=list(modality_config.get("table_keywords", [])),
        modality_figure_keywords=list(modality_config.get("figure_keywords", [])),
        modality_audio_keywords=list(modality_config.get("audio_keywords", [])),
        modality_table_boost_multiplier=float(modality_config.get("table_boost_multiplier", 0.30)),
        modality_extra_pass=bool(modality_config.get("extra_pass", True)),
        modality_filtered_pass_limit_ratio=float(modality_config.get("filtered_pass_limit_ratio", 0.5)),
        vlm_list_caption_score_penalty=float(modality_config.get("vlm_list_caption_score_penalty", 0.40)),
        router_agentic_confidence_threshold=float(
            retrieval_config.get("routing", {}).get("agentic_confidence_threshold", 0.55)
        ),
        slec_enabled=env_bool("SLEC_ENABLED", slec_config.get("enabled", True)),
        slec_supported_threshold=float(slec_config.get("supported_threshold", 0.55)),
        slec_partial_threshold=float(slec_config.get("partial_threshold", 0.30)),
        slec_refuse_below=float(slec_config.get("refuse_below", 0.40)),
        slec_drop_unsupported=bool(slec_config.get("drop_unsupported", True)),
        slec_min_sentence_chars=int(slec_config.get("min_sentence_chars", 12)),
        slec_max_sentences=int(slec_config.get("max_sentences", 24)),
        slec_skip_routes=list(slec_config.get("skip_routes", ["claim_check"])),
        quality_gate_conf_caution=float(quality_gate_config.get("conf_caution", 0.50)),
        quality_gate_conf_fail=float(quality_gate_config.get("conf_fail", 0.30)),
        quality_gate_slec_caution=float(quality_gate_config.get("slec_caution", 0.60)),
        quality_gate_slec_fail=float(quality_gate_config.get("slec_fail", 0.40)),
        quality_gate_citation_caution=float(quality_gate_config.get("citation_caution", 0.80)),
        quality_gate_citation_fail=float(quality_gate_config.get("citation_fail", 0.50)),
        visual_caption_fallback_enabled=bool(visual_guardrails_config.get("caption_fallback_enabled", True)),
        graph_probe_enabled=bool(graph_probe_config.get("enabled", True)),
        graph_probe_min_entities=int(graph_probe_config.get("min_entities", 2)),
        graph_probe_top_chunks_to_scan=int(graph_probe_config.get("top_chunks_to_scan", 5)),
        graph_probe_max_graph_chunks=int(graph_probe_config.get("max_graph_chunks", 4)),
        graph_probe_skip_routes=list(graph_probe_config.get("skip_routes", ["claim_check"])),
        min_handwriting_quality_score=refusal_config.get("min_handwriting_quality_score", 0.72),
        min_handwriting_confidence=refusal_config.get("min_handwriting_confidence", 0.8),
        min_blur_variance=image_quality_config.get("min_blur_variance", 80.0),
        min_brightness=image_quality_config.get("min_brightness", 45.0),
        max_brightness=image_quality_config.get("max_brightness", 230.0),
        min_contrast=image_quality_config.get("min_contrast", 18.0),
        max_abs_skew_degrees=image_quality_config.get("max_abs_skew_degrees", 12.0),
        audio_whisper_model=audio_config.get("whisper_model", "small"),
        audio_whisper_device=env_value("AUDIO_WHISPER_DEVICE", audio_config.get("whisper_device", "cpu")),
        audio_whisper_compute_type=env_value(
            "AUDIO_WHISPER_COMPUTE_TYPE", audio_config.get("whisper_compute_type", "int8")
        ),
        audio_whisper_beam_size=int(audio_config.get("whisper_beam_size", 5)),
        audio_whisper_vad_filter=bool(audio_config.get("whisper_vad_filter", True)),
        audio_chunk_target_seconds=float(audio_config.get("audio_chunk_target_seconds", 45.0)),
        audio_chunk_min_seconds=float(audio_config.get("audio_chunk_min_seconds", 10.0)),
        ocr_text_detection_model_name=ocr_config.get("text_detection_model_name", "PP-OCRv5_mobile_det"),
        ocr_text_recognition_model_name=ocr_config.get("text_recognition_model_name"),
        ocr_text_det_limit_side_len=int(ocr_config.get("text_det_limit_side_len", 1280)),
        ocr_text_det_limit_type=ocr_config.get("text_det_limit_type", "max"),
        ocr_rec_score_threshold=float(ocr_config.get("rec_score_threshold", 0.5)),
        ocr_enable_grayscale_variant=str(ocr_config.get("enable_grayscale_variant", "auto")).lower(),
        ocr_grayscale_trigger_confidence=float(ocr_config.get("grayscale_trigger_confidence", 0.85)),
        ocr_recognition_engine=str(ocr_config.get("recognition_engine", "easyocr")).lower(),
        ocr_vietocr_model_name=str(ocr_config.get("vietocr_model_name", "vgg_transformer")),
        ocr_vietocr_device=env_value("OCR_VIETOCR_DEVICE", ocr_config.get("vietocr_device", "cpu")),
        ocr_easyocr_gpu=env_bool("OCR_EASYOCR_GPU", bool(ocr_config.get("easyocr_gpu", False))),
        ocr_garbled_vi_diacritic_ratio=float(ocr_config.get("garbled_vi_diacritic_ratio", 0.03)),
        image_ocr_text_word_threshold=int(ocr_config.get("image_ocr_text_word_threshold", 20)),
        image_numeric_ratio_vlm_trigger=float(ocr_config.get("image_numeric_ratio_vlm_trigger", 0.30)),
        image_structure_vlm_max_side_px=int(ocr_config.get("image_structure_vlm_max_side_px", 2048)),
        image_structure_vlm_min_digit_cell_ratio=float(ocr_config.get("image_structure_vlm_min_digit_cell_ratio", 0.15)),
        image_structure_vlm_num_predict=int(ocr_config.get("image_structure_vlm_num_predict", 1536)),
        layout_column_detection_enabled=bool(ocr_config.get("layout_column_detection_enabled", True)),
        layout_column_min_gutter_ratio=float(ocr_config.get("layout_column_min_gutter_ratio", 0.04)),
        layout_column_min_band_occupancy_ratio=float(ocr_config.get("layout_column_min_band_occupancy_ratio", 0.30)),
        vlm_table_fallback_enabled=bool(ocr_config.get("vlm_table_fallback_enabled", True)),
        vlm_table_confidence=float(ocr_config.get("vlm_table_confidence", 0.5)),
        pdf_render_scale=float(pdf_config.get("render_scale", 1.5)),
        visual_embedding_enabled=env_bool(
            "VISUAL_EMBEDDING_ENABLED", visual_config.get("enabled", False)
        ),
        visual_embedding_model=visual_config.get(
            "model", "google/siglip-base-patch16-224"
        ),
        visual_embedding_device=env_value(
            "VISUAL_EMBEDDING_DEVICE", visual_config.get("device", "cpu")
        ),
        visual_embedding_dense_size=int(visual_config.get("dense_size", 768)),
        visual_embedding_batch_size=int(
            env_value(
                "VISUAL_EMBEDDING_BATCH_SIZE", visual_config.get("batch_size", 4)
            )
        ),
        visual_embedding_backend=visual_config.get("embedding_backend", "pytorch"),
        qdrant_visual_collection_name=qdrant_config.get(
            "visual_collection_name", "agentbook_visual"
        ),
        visual_retrieval_top_k=int(retrieval_config.get("visual_retrieval_top_k", 3)),
        lexical_fallback_multiplier=int(retrieval_section.get("lexical_fallback_multiplier", 3)),
        chunk_min_content_chars=int(chunking_config.get("min_content_chars", 50)),
        figure_captioner_max_workers=int(figure_captioner_config.get("max_workers", 4)),
        ollama_caption_timeout_seconds=float(figure_captioner_config.get("ollama_timeout_seconds", 300.0)),
        figure_image_max_side_px=int(figure_captioner_config.get("image_max_side_px", 1024)),
        vlm_caption_num_ctx=int(figure_captioner_config.get("num_ctx", 8192)),
        caption_list_bullet_ratio_max=float(figure_captioner_config.get("list_bullet_ratio_max", 0.70)),
        caption_list_bullet_max_words=int(figure_captioner_config.get("list_bullet_max_words", 3)),
        memory_max_turns=int(memory_config.get("max_turns", 10)),
        self_rag_reflection_enabled=env_bool(
            "SELF_RAG_REFLECTION_ENABLED", llm_config.get("self_rag_reflection_enabled", False)
        ),
        graph_fuzzy_dedup_threshold=float(graph_config.get("fuzzy_dedup_threshold", 0.88)),
        graph_semantic_dedup_threshold=float(graph_config.get("semantic_dedup_threshold", 0.82)),
        graph_display_min_confidence=float(graph_config.get("display_min_confidence", 0.5)),
        graph_display_min_mentions=int(graph_config.get("display_min_mentions", 2)),
        graph_max_entities_fetch=int(graph_config.get("max_entities_fetch", 400)),
        graph_max_relations_fetch=int(graph_config.get("max_relations_fetch", 400)),
        graph_max_visible_nodes=int(graph_config.get("max_visible_nodes", 50)),
        graph_focus_primary_cap=int(graph_config.get("focus_primary_cap", 15)),
        graph_focus_neighbor_cap=int(graph_config.get("focus_neighbor_cap", 12)),
        graph_focus_fallback_cap=int(graph_config.get("focus_fallback_cap", 20)),
        graph_mindmap_entity_fetch=int(graph_config.get("mindmap_entity_fetch", 160)),
        graph_mindmap_entity_cap=int(graph_config.get("mindmap_entity_cap", 60)),
        graph_semantic_relation_max_concepts=int(graph_semrel_config.get("max_concepts", 25)),
        graph_semantic_relation_max_passages=int(graph_semrel_config.get("max_passages", 18)),
        graph_semantic_relation_max_passage_chars=int(graph_semrel_config.get("max_passage_chars", 600)),
        graph_semantic_relation_gleaning=bool(graph_semrel_config.get("gleaning", False)),
        graph_min_knowledge_edges_for_no_fallback=int(graph_semrel_config.get("min_knowledge_edges_for_no_fallback", 3)),
        extraction_mode=str(extraction_config.get("mode", "dynamic")).lower(),
        extraction_default_entity_types=list(
            extraction_config.get(
                "default_entity_types",
                ["concept", "person", "organization", "location",
                 "event", "artifact", "time", "quantity"],
            )
        ),
        extraction_output_format=str(extraction_config.get("output_format", "json")).lower(),
        extraction_max_gleanings=int(extraction_config.get("max_gleanings", 0)),
        extraction_domain_hint_field=str(extraction_config.get("domain_hint_field", "subject")),
        extraction_few_shots=list(extraction_config.get("few_shots", [])),
        extraction_entity_backend=str(extraction_config.get("entity_backend", "llm")),
        extraction_gliner_model=str(extraction_config.get("gliner_model", "gliner-community/gliner_medium-v2.5")),
        extraction_gliner_threshold=float(extraction_config.get("gliner_threshold", 0.5)),
        extraction_ontology=dict(extraction_config.get("ontology", {})),
        extraction_graph_entity_types=list(extraction_config.get("graph_entity_types", [])),
        extraction_relation_types=list(extraction_config.get("relation_types", [])),
        extraction_entity_type_descriptions=dict(extraction_config.get("entity_type_descriptions", {})),
        extraction_entity_refine_enabled=bool(extraction_config.get("entity_refine_enabled", True)),
        extraction_entity_refine_max_entities=int(extraction_config.get("entity_refine_max_entities", 200)),
        extraction_excluded_sections=list(extraction_config.get("excluded_sections", [])),
        extraction_merge_threshold=float(extraction_config.get("merge_threshold", 0.82)),
        extraction_cross_lingual_merge_threshold=float(extraction_config.get("cross_lingual_merge_threshold", 0.74)),
        viz_config=dict(viz_config),
        # ── New fields ──────────────────────────────────────────────────────
        # Inference engine tuning
        inference_default_answer_language=inference_cfg.get("default_answer_language", "vi"),
        inference_default_chitchat_answer=inference_cfg.get("default_chitchat_answer", "Xin chào! Tôi có thể giúp gì cho bạn?"),
        inference_graph_timeout_seconds=float(inference_cfg.get("graph_timeout_seconds", 25.0)),
        inference_visual_inline_limit=int(inference_cfg.get("visual_inline_retrieval_limit", 4)),
        inference_visual_inline_max=int(inference_cfg.get("visual_inline_max", 2)),
        inference_retry_evidence_snippet_chars=int(inference_cfg.get("retry_evidence_snippet_chars", 300)),
        inference_fallback_snippet_chars=int(inference_cfg.get("fallback_snippet_chars", 200)),
        inference_retry_evidence_limit=int(inference_cfg.get("retry_evidence_limit", 5)),
        inference_graph_priority_score_boost=float(inference_cfg.get("graph_priority_score_boost", 0.25)),
        inference_graph_boost_score=float(inference_cfg.get("graph_boost_score", 0.10)),
        inference_min_chunk_chars=int(inference_cfg.get("min_chunk_chars", 40)),
        inference_multi_doc_min_sources=int(inference_cfg.get("multi_doc_min_sources", 3)),
        inference_synthesis_route_types=list(inference_cfg.get("synthesis_route_types", ["summarization", "comparison", "graph_relation"])),
        inference_self_rag_evidence_char_limit=int(inference_cfg.get("self_rag_evidence_char_limit", 3000)),
        inference_self_rag_answer_char_limit=int(inference_cfg.get("self_rag_answer_char_limit", 2000)),
        inference_memory_context_header=inference_cfg.get("memory_context_header", "LỊCH SỬ LIÊN QUAN"),
        inference_language_names=dict(inference_cfg.get("language_names", {
            "vi": "tiếng Việt", "en": "English", "zh": "Chinese",
            "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
        })),
        inference_route_prompt_map=dict(inference_cfg.get("route_prompt_map", {
            "summarization": "summarization.txt", "comparison": "comparison.txt",
            "claim_check": "claim_check.txt", "graph_relation": "graph_relation.txt",
        })),
        inference_multi_source_prompt_file=inference_cfg.get("multi_source_prompt_file", "multi_source.txt"),
        inference_default_prompt_file=inference_cfg.get("default_prompt_file", "qa_grounded.txt"),
        inference_substantive_chunk_filter_prefixes=list(inference_cfg.get("substantive_chunk_filter_prefixes", [
            "Ghi nhớ", "Note:", "Starter Pack", "Tập trung", "Người mới",
            "nên bắt đầu", "scikit-learn User Guide", "Dive into Deep",
            "Xem thêm tại", "Link:", "URL:",
        ])),
        vlm_query_enabled=bool(model_config.get("vlm_query", {}).get("enabled", True)),
        vlm_query_provider=str(model_config.get("vlm_query", {}).get("provider", "auto")),
        vlm_query_ollama_model=str(model_config.get("vlm_query", {}).get("ollama_model", "")),
        vlm_query_openai_model=str(model_config.get("vlm_query", {}).get("openai_model", "")),
        vlm_query_max_images=int(model_config.get("vlm_query", {}).get("max_images", 2)),
        vlm_query_image_max_side_px=int(model_config.get("vlm_query", {}).get("image_max_side_px", 1024)),
        vlm_query_timeout_seconds=float(model_config.get("vlm_query", {}).get("timeout_seconds", 300.0)),
        vlm_query_caption_fallback=bool(model_config.get("vlm_query", {}).get("caption_fallback", True)),
        vlm_query_verify_enabled=bool(model_config.get("vlm_query", {}).get("verify_enabled", True)),
        vlm_query_verify_refuse_confidence=float(model_config.get("vlm_query", {}).get("verify_refuse_confidence", 0.75)),
        # Docling
        docling_queue_max_size=int(docling_cfg.get("queue_max_size", 1)),
        docling_images_scale=float(docling_cfg.get("images_scale", 0.5)),
        docling_subprocess_enabled=bool(docling_cfg.get("subprocess_enabled", True)),
        docling_subprocess_timeout_seconds=float(docling_cfg.get("subprocess_timeout_seconds", 900.0)),
        free_ollama_before_parse=env_bool("FREE_OLLAMA_BEFORE_PARSE", bool(docling_cfg.get("free_ollama_before_parse", True))),
        # Structural KG linker (belongs_to section edges)
        structure_linker_enabled=bool(structure_linker_cfg.get("enabled", True)),
        structure_section_confidence=float(structure_linker_cfg.get("section_confidence", 0.9)),
        structure_belongs_to_confidence=float(structure_linker_cfg.get("belongs_to_confidence", 0.85)),
        # Messages
        messages_refusal_answer=dict(messages_config.get("refusal_answer", {
            "vi": "Tôi không tìm thấy đủ bằng chứng trong tài liệu được cung cấp để trả lời câu hỏi này.",
            "en": "I could not find sufficient evidence in the provided documents to answer this question.",
        })),
        messages_refusal_prefix=dict(messages_config.get("refusal_prefix", {
            "vi": "Tôi không tìm thấy đủ bằng chứng",
            "en": "I could not find sufficient evidence",
        })),
        messages_low_confidence_warning=dict(messages_config.get("low_confidence_warning", {
            "vi": "\n\n> Cảnh báo: Một số nhận định chưa được bằng chứng hỗ trợ trực tiếp.",
            "en": "\n\n> Warning: Some claims may not be directly supported by the evidence.",
        })),
        messages_partial_confidence_warning=dict(messages_config.get("partial_confidence_warning", {
            "vi": "\n\n> ⚠️ Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra lại nguồn gốc.",
            "en": "\n\n> ⚠️ This answer is based on limited-confidence evidence. Please verify the sources.",
        })),
        messages_insufficient_evidence_refusal=dict(messages_config.get("insufficient_evidence_refusal", {
            "vi": "Không đủ bằng chứng đáng tin cậy để tạo câu trả lời có citation hợp lệ.",
            "en": "Insufficient reliable evidence to produce a validly-cited answer.",
        })),
        messages_self_rag_unsupported_prefix=dict(messages_config.get("self_rag_unsupported_prefix", {
            "vi": "⚠️ Chưa có đủ bằng chứng",
            "en": "⚠️ Insufficient evidence",
        })),
        # Agentic critic
        agentic_critic_max_follow_ups=int(retrieval_section.get("agentic_critic_max_follow_ups", 2)),
        agentic_critic_context_buffer=int(retrieval_section.get("agentic_critic_context_buffer", 4)),
        # Extraction heuristics
        extraction_max_chars_per_llm_batch=int(heuristics_cfg.get("max_chars_per_llm_batch", 3000)),
        extraction_max_entity_words=int(heuristics_cfg.get("max_entity_words", 7)),
        extraction_stopword_only_max_words=int(heuristics_cfg.get("stopword_only_max_words", 2)),
        extraction_min_confidence=float(heuristics_cfg.get("min_extraction_confidence", 0.5)),
        extraction_merge_confidence_boost=float(heuristics_cfg.get("merge_confidence_boost", 0.02)),
        extraction_merge_confidence_max=float(heuristics_cfg.get("merge_confidence_max", 0.97)),
        extraction_confidence_method_keyword=float(heuristics_cfg.get("confidence_method_keyword", 0.78)),
        extraction_confidence_metric=float(heuristics_cfg.get("confidence_metric", 0.72)),
        extraction_confidence_capitalized_term=float(heuristics_cfg.get("confidence_capitalized_term", 0.55)),
        extraction_confidence_vietnamese_ner=float(heuristics_cfg.get("confidence_vietnamese_ner", 0.68)),
        extraction_method_keywords=method_keywords_cfg,
        extraction_metric_terms=metric_terms_cfg,
        extraction_en_stopwords=en_stopwords_cfg,
        extraction_vi_stopwords=vi_stopwords_cfg,
        extraction_compound_heads=compound_heads_cfg,
        extraction_anaphora_pattern=anaphora_pattern_cfg,
        extraction_structural_junk_patterns=structural_junk_patterns_cfg,
        # OCR preprocessing
        ocr_preprocessing_min_long_edge_px=int(ocr_preproc_cfg.get("min_long_edge_px", 1600)),
        ocr_preprocessing_low_contrast_threshold=float(ocr_preproc_cfg.get("low_contrast_threshold", 60.0)),
        ocr_preprocessing_denoise_h=int(ocr_preproc_cfg.get("denoise_h", 8)),
        ocr_preprocessing_denoise_template_window=int(ocr_preproc_cfg.get("denoise_template_window", 7)),
        ocr_preprocessing_denoise_search_window=int(ocr_preproc_cfg.get("denoise_search_window", 21)),
        ocr_preprocessing_clahe_clip_limit=float(ocr_preproc_cfg.get("clahe_clip_limit", 2.5)),
        ocr_preprocessing_clahe_tile_grid=list(ocr_preproc_cfg.get("clahe_tile_grid", [8, 8])),
        ocr_preprocessing_unsharp_sigma=float(ocr_preproc_cfg.get("unsharp_sigma", 2.0)),
        ocr_preprocessing_unsharp_alpha=float(ocr_preproc_cfg.get("unsharp_alpha", 1.4)),
        ocr_preprocessing_unsharp_beta=float(ocr_preproc_cfg.get("unsharp_beta", -0.4)),
        ocr_preprocessing_adaptive_block_size=int(ocr_preproc_cfg.get("adaptive_block_size", 25)),
        ocr_preprocessing_adaptive_c=int(ocr_preproc_cfg.get("adaptive_c", 10)),
        ocr_preprocessing_line_tolerance_px=int(ocr_preproc_cfg.get("line_tolerance_px", 15)),
        ocr_preprocessing_dedup_jaccard_threshold=float(ocr_preproc_cfg.get("dedup_jaccard_threshold", 0.80)),
        ocr_preprocessing_dedup_subset_coverage_threshold=float(ocr_preproc_cfg.get("dedup_subset_coverage_threshold", 0.85)),
        ocr_preprocessing_dedup_confidence_boost_margin=float(ocr_preproc_cfg.get("dedup_confidence_boost_margin", 0.05)),
        ocr_easyocr_language_map=dict(ocr_config.get("easyocr_language_map", {"vi": ["vi"], "en": ["en"]})),
    )
