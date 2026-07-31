from __future__ import annotations

import re
from uuid import NAMESPACE_URL, uuid5

from src.models.material import BoundingBox, MaterialBlock, MaterialPage
from src.processing.column_segmenter import annotate_columns
from src.processing.language_detector import detect_block_language, detect_document_language
from src.processing.reading_order import order_blocks_by_reading
from src.processing.types import BBox, BlockType, ParsedBlock, ParsedDocument, ParsedPage

_BULLET_PREFIX = re.compile(r"^[\u25a1\u2022\u25cf\u25aa\u25ab\u2013\u2014\-\*\u25c6\u25c7\u25cb]\s")
_TEXT_FRAGMENT_TYPES = {BlockType.PARAGRAPH.value, BlockType.LIST.value}


class LayoutNormalizer:
    def __init__(
        self,
        *,
        column_detection_enabled: bool = True,
        column_min_gutter_ratio: float = 0.04,
        column_min_band_occupancy_ratio: float = 0.30,
    ) -> None:
        self.column_detection_enabled = column_detection_enabled
        self.column_min_gutter_ratio = column_min_gutter_ratio
        self.column_min_band_occupancy_ratio = column_min_band_occupancy_ratio

    def normalize(self, parsed: ParsedDocument) -> ParsedDocument:
        normalized_pages: list[ParsedPage] = []
        declared_lang = parsed.language if parsed.language not in ("unknown", "") else None
        all_texts = [block.content for page in parsed.pages for block in page.blocks if block.content.strip()]
        doc_lang = declared_lang or detect_document_language(all_texts, fallback="unknown")

        for page in sorted(parsed.pages, key=lambda item: item.page_number):
            # Keep figure blocks even when empty — content filled by FigureCaptioner downstream.
            blocks = [
                block for block in page.blocks
                if block.content.strip() or block.block_type == BlockType.FIGURE.value
            ]
            # Geometry-aware order: a naive `bbox.y1` ascending sort reverses every
            # PDF (Docling uses a bottom-left origin where the top edge has the
            # larger y) and scrambles multi-column pages. order_blocks_by_reading
            # keeps the parser's reading_order (which handles columns) unless the
            # page is grossly reversed, in which case it falls back to geometry.
            blocks = order_blocks_by_reading(blocks)
            # Column/panel-aware annotation for OCR images: detect column bands and
            # stamp column_index (order preserved) so the line merger below never
            # merges across a column/panel boundary. No-op (single column) for
            # normal pages; keeps genuine table rows together (row-major).
            if self.column_detection_enabled:
                blocks = annotate_columns(
                    blocks,
                    min_gutter_ratio=self.column_min_gutter_ratio,
                    min_band_occupancy_ratio=self.column_min_band_occupancy_ratio,
                )
            blocks = self._merge_ocr_lines(blocks)
            blocks = self._merge_text_fragments(blocks)
            normalized_blocks = [
                block.model_copy(
                    update={
                        "block_id": block.block_id or self._block_id(parsed.source_path, page.page_number, index, block.content),
                        "block_index": index,
                        "block_type": self._normalize_block_type(block),
                        "content": self._normalize_text(
                            block.content,
                            preserve_newlines=block.block_type == BlockType.TABLE.value,
                        ),
                        "language": detect_block_language(block.content, fallback=doc_lang),
                        "reading_order": index,
                    }
                )
                for index, block in enumerate(blocks)
            ]
            normalized_pages.append(page.model_copy(update={"blocks": normalized_blocks}))
        return parsed.model_copy(update={"pages": normalized_pages, "language": doc_lang})

    @staticmethod
    def to_material_pages(parsed: ParsedDocument) -> list[MaterialPage]:
        return [
            MaterialPage(
                page_number=page.page_number,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
                ocr_confidence=page.ocr_confidence,
                blocks=[
                    MaterialBlock(
                        block_id=block.block_id,
                        block_index=block.block_index,
                        block_type=block.block_type,
                        content=block.content,
                        language=block.language,
                        bbox=LayoutNormalizer._to_material_bbox(block.bbox),
                        ocr_confidence=block.ocr_confidence,
                        reading_order=block.reading_order,
                        extra=block.extra,
                    )
                    for block in page.blocks
                ],
            )
            for page in parsed.pages
        ]

    @staticmethod
    def _merge_ocr_lines(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        if not blocks:
            return blocks

        ocr_with_bbox = [block for block in blocks if block.block_type == BlockType.OCR_TEXT.value and block.bbox]
        if len(ocr_with_bbox) < 3 or len(ocr_with_bbox) < len(blocks) * 0.6:
            return blocks

        heights = sorted(block.bbox.y2 - block.bbox.y1 for block in ocr_with_bbox if block.bbox)
        median_height = heights[len(heights) // 2]
        gap_threshold = median_height * 0.8

        groups: list[list[ParsedBlock]] = []
        current_group: list[ParsedBlock] = []
        for block in blocks:
            if not current_group:
                current_group.append(block)
                continue
            previous = current_group[-1]
            can_merge = (
                block.block_type == BlockType.OCR_TEXT.value
                and previous.block_type == BlockType.OCR_TEXT.value
                and block.bbox is not None
                and previous.bbox is not None
                and (block.bbox.y1 - previous.bbox.y2) <= gap_threshold
                # Never merge across a detected column/panel boundary. Single-column
                # pages carry no column_index (None == None) → unchanged behavior.
                and block.extra.get("column_index") == previous.extra.get("column_index")
            )
            if can_merge:
                current_group.append(block)
            else:
                groups.append(current_group)
                current_group = [block]

        if current_group:
            groups.append(current_group)

        merged: list[ParsedBlock] = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
                continue
            all_bboxes = [block.bbox for block in group if block.bbox]
            merged_bbox = (
                BBox(
                    x1=min(bbox.x1 for bbox in all_bboxes),
                    y1=min(bbox.y1 for bbox in all_bboxes),
                    x2=max(bbox.x2 for bbox in all_bboxes),
                    y2=max(bbox.y2 for bbox in all_bboxes),
                )
                if all_bboxes
                else None
            )
            confidences = [block.ocr_confidence for block in group if block.ocr_confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            merged.append(
                group[0].model_copy(
                    update={
                        "content": " ".join(block.content for block in group),
                        "bbox": merged_bbox,
                        "ocr_confidence": avg_confidence,
                        "extra": {**group[0].extra, "merged_line_count": len(group)},
                    }
                )
            )
        return merged

    @staticmethod
    def _merge_text_fragments(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        if not blocks:
            return blocks
        merged: list[ParsedBlock] = []
        for block in blocks:
            if not merged:
                merged.append(block)
                continue
            previous = merged[-1]
            if LayoutNormalizer._should_merge_text_fragment(previous, block):
                separator = "" if block.content.strip() in {":", ";", ","} else " "
                merged[-1] = previous.model_copy(
                    update={
                        "content": f"{previous.content.rstrip()}{separator}{block.content.lstrip()}",
                        "extra": {
                            **previous.extra,
                            "merged_fragment_count": int(previous.extra.get("merged_fragment_count", 1)) + 1,
                        },
                    }
                )
            else:
                merged.append(block)
        return merged

    @staticmethod
    def _should_merge_text_fragment(previous: ParsedBlock, current: ParsedBlock) -> bool:
        if previous.page_number != current.page_number:
            return False
        if previous.bbox is not None or current.bbox is not None:
            return False
        if previous.block_type not in _TEXT_FRAGMENT_TYPES or current.block_type not in _TEXT_FRAGMENT_TYPES:
            return False
        prev_text = previous.content.strip()
        curr_text = current.content.strip()
        if not prev_text or not curr_text:
            return False
        if curr_text in {":", ";", ","}:
            return True
        if prev_text.endswith(":"):
            return True
        return len(prev_text) <= 28 and not prev_text.endswith((".", "?", "!", ";"))

    @staticmethod
    def _to_material_bbox(bbox: BBox | None) -> BoundingBox | None:
        if bbox is None:
            return None
        return BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)

    @staticmethod
    def _normalize_text(text: str, *, preserve_newlines: bool = False) -> str:
        cleaned = text.replace("\x00", " ")
        if not preserve_newlines:
            return " ".join(cleaned.split())
        lines = [" ".join(line.split()) for line in cleaned.splitlines()]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _normalize_block_type(block: ParsedBlock) -> str:
        text = block.content.strip()
        current = block.block_type
        # Keep explicit parser-assigned types \u2014 only normalise raw ocr_text blocks
        if current in {item.value for item in BlockType} and current != BlockType.OCR_TEXT.value:
            return current
        # Table: pipe-delimited markdown produced by spreadsheet parser or docling
        if "|" in text and "\n" in text:
            return BlockType.TABLE.value
        # List: bullet-prefixed lines
        if _BULLET_PREFIX.match(text) or text.startswith(("-", "*", "\u2022")):
            return BlockType.LIST.value
        # Heading heuristic for OCR fragments:
        # Very short text (\u2264 40 chars) without trailing sentence punctuation \u2192 heading.
        # Longer text (41-80 chars) only if it looks title-cased (works for EN, not VI).
        if len(text) <= 80 and not text.endswith((".", ";", ",")) and "," not in text:
            if len(text) <= 40:
                return BlockType.HEADING.value
            words = text.split()
            if words and sum(w[0].isupper() for w in words if w) / len(words) >= 0.5:
                return BlockType.HEADING.value
        return BlockType.PARAGRAPH.value

    @staticmethod
    def _block_id(source_path: str, page_number: int, index: int, content: str) -> str:
        return f"blk-{uuid5(NAMESPACE_URL, f'{source_path}:{page_number}:{index}:{content[:80]}').hex[:12]}"

