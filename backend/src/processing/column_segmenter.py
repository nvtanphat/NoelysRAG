"""Column/panel-aware layout segmentation for OCR images.

EasyOCR emits text blocks roughly row-major. On multi-panel pages — financial
dashboards, side-by-side charts — the line merger (which groups by vertical gap
only) then collapses blocks from *different* columns that happen to share a row
into one block, scrambling their values left-to-right (e.g. a left chart's
numbers glued to a right chart's, glued to a table's cells).

This module detects column bands from vertical-whitespace gutters and stamps
each block with ``extra["column_index"]`` so the line merger never merges across
a column boundary. It deliberately does NOT re-order the blocks: keeping the
original row-major order means a genuine multi-column TABLE (where a row's label
and values live in different columns) stays row-adjacent and lands in one chunk,
while INDEPENDENT panels are simply kept from merging into each other.

Design constraints:
  * Generic — every threshold is a ratio of the page's own geometry; no hardcoded
    pixel coordinates or column counts.
  * Conservative — a single-column page has no interior gutter, so detection
    returns None and stamping is a no-op (zero regression).
  * Scoped — detection keys off OCR_TEXT blocks, so PDF/docling pages (whose
    blocks already carry column-correct reading_order) are left untouched.
"""

from __future__ import annotations

from src.processing.types import BlockType, ParsedBlock

# A block wider than this fraction of the content width spans multiple columns
# (a full-width title/header) — it is an anchor, not a member of any one column.
_FULL_WIDTH_RATIO = 0.6
_MIN_OCR_BLOCKS = 3


def _positioned_ocr_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    return [
        b
        for b in blocks
        if b.block_type == BlockType.OCR_TEXT.value and b.bbox is not None
    ]


def detect_columns(
    blocks: list[ParsedBlock],
    *,
    min_gutter_ratio: float,
    min_band_occupancy_ratio: float,
) -> list[tuple[float, float]] | None:
    """Detect column bands as ``[(x_min, x_max), ...]`` or None for single-column.

    Builds an x-occupancy projection (weighted by each block's vertical span so a
    gutter is empty across most of the page height, not just one row), then keeps
    interior empty runs wider than ``min_gutter_ratio * content_w`` as gutters.
    The bands between gutters are the columns. Returns None when there are too few
    OCR blocks or no interior gutter (i.e. a single column).
    """
    ocr_blocks = _positioned_ocr_blocks(blocks)
    if len(ocr_blocks) < _MIN_OCR_BLOCKS:
        return None

    x_min = min(b.bbox.x1 for b in ocr_blocks)
    x_max = max(b.bbox.x2 for b in ocr_blocks)
    content_w = x_max - x_min
    if content_w <= 0:
        return None

    heights = sorted(b.bbox.y2 - b.bbox.y1 for b in ocr_blocks)
    median_height = heights[len(heights) // 2] or 1.0
    content_h = max(b.bbox.y2 for b in ocr_blocks) - min(b.bbox.y1 for b in ocr_blocks)
    if content_h <= 0:
        return None

    # Bin the x-axis at the text scale (median line height); bound the count.
    bin_width = max(median_height, content_w / 2000.0)
    n_bins = max(1, min(2000, int(content_w / bin_width) + 1))
    occupancy = [0.0] * n_bins

    for b in ocr_blocks:
        height = b.bbox.y2 - b.bbox.y1
        start = int((b.bbox.x1 - x_min) / content_w * n_bins)
        end = int((b.bbox.x2 - x_min) / content_w * n_bins)
        start = max(0, min(n_bins - 1, start))
        end = max(0, min(n_bins - 1, end))
        for i in range(start, end + 1):
            occupancy[i] += height

    # Normalise to a fraction of page height; cap at 1 (stacked blocks over-count).
    filled = [min(1.0, occ / content_h) >= min_band_occupancy_ratio for occ in occupancy]

    # Trim leading/trailing empties — those are page margins, not interior gutters.
    first = next((i for i, f in enumerate(filled) if f), None)
    last = next((i for i in range(n_bins - 1, -1, -1) if filled[i]), None)
    if first is None or last is None or first >= last:
        return None

    min_gutter_bins = max(1, int((min_gutter_ratio * content_w) / bin_width))
    columns: list[tuple[float, float]] = []
    band_start_bin = first
    run = 0
    i = first
    while i <= last:
        if filled[i]:
            if run >= min_gutter_bins and i - run > band_start_bin:
                # Close the band that ended before this gutter.
                columns.append(
                    (
                        x_min + band_start_bin / n_bins * content_w,
                        x_min + (i - run) / n_bins * content_w,
                    )
                )
                band_start_bin = i
            run = 0
        else:
            run += 1
        i += 1
    # Final band.
    columns.append(
        (
            x_min + band_start_bin / n_bins * content_w,
            x_min + (last + 1) / n_bins * content_w,
        )
    )

    if len(columns) < 2:
        return None
    return columns


def _column_index(bbox, columns: list[tuple[float, float]]) -> int:
    """Index of the column whose band contains the block's horizontal centre.

    Falls back to the nearest column when the centre lands inside a gutter.
    """
    cx = (bbox.x1 + bbox.x2) / 2.0
    for idx, (lo, hi) in enumerate(columns):
        if lo <= cx <= hi:
            return idx
    return min(
        range(len(columns)),
        key=lambda idx: abs(cx - (columns[idx][0] + columns[idx][1]) / 2.0),
    )


def stamp_columns(
    blocks: list[ParsedBlock],
    columns: list[tuple[float, float]],
    *,
    full_width_ratio: float = _FULL_WIDTH_RATIO,
) -> list[ParsedBlock]:
    """Stamp ``extra['column_index']`` on each block, preserving input order.

    The reading order is intentionally left unchanged so a genuine multi-column
    table keeps its rows together; the stamp exists only so the downstream line
    merger refuses to merge across a column/panel boundary. Full-width blocks
    (titles spanning columns) and bbox-less blocks (the VLM structure overview,
    figures) get no column_index — they remain mergeable/neutral anchors.
    """
    content_w = columns[-1][1] - columns[0][0]
    result: list[ParsedBlock] = []
    for b in blocks:
        is_full_width = (
            b.bbox is not None
            and content_w > 0
            and (b.bbox.x2 - b.bbox.x1) >= full_width_ratio * content_w
        )
        if b.bbox is None or is_full_width:
            result.append(b)
            continue
        idx = _column_index(b.bbox, columns)
        result.append(b.model_copy(update={"extra": {**b.extra, "column_index": idx}}))
    return result


def annotate_columns(
    blocks: list[ParsedBlock],
    *,
    min_gutter_ratio: float,
    min_band_occupancy_ratio: float,
) -> list[ParsedBlock]:
    """Detect columns and stamp column_index (order preserved); no-op if single-column."""
    columns = detect_columns(
        blocks,
        min_gutter_ratio=min_gutter_ratio,
        min_band_occupancy_ratio=min_band_occupancy_ratio,
    )
    if not columns:
        return blocks
    return stamp_columns(blocks, columns)
