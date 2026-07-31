from __future__ import annotations

from src.processing.column_segmenter import (
    annotate_columns,
    detect_columns,
    stamp_columns,
)
from src.processing.types import BBox, BlockType, ParsedBlock


def _ocr(block_id: str, *, x1: float, x2: float, y1: float, y2: float) -> ParsedBlock:
    return ParsedBlock(
        block_id=block_id,
        block_index=0,
        block_type=BlockType.OCR_TEXT.value,
        content=block_id,
        page_number=1,
        language="vi",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        ocr_confidence=0.95,
        reading_order=0,
        source="easyocr",
    )


def _single_column_blocks() -> list[ParsedBlock]:
    # Wrapped text lines stacked vertically, all sharing one x-band.
    return [
        _ocr(f"line{i}", x1=10, x2=600, y1=i * 30, y2=i * 30 + 20)
        for i in range(8)
    ]


def _three_panel_blocks() -> list[ParsedBlock]:
    """3 columns (x 0-300 / 360-660 / 720-1000) with a wide gutter between each.

    Each column is dense top-to-bottom so its x-bins are 'filled' while the
    gutters stay empty. Blocks are listed ROW-MAJOR (the order EasyOCR emits) to
    prove the segmenter re-orders them column-major.
    """
    rows_y = [(i * 90, i * 90 + 40) for i in range(8)]  # content height ~ 760
    blocks: list[ParsedBlock] = []
    cols = {"L": (20, 280), "M": (380, 640), "R": (740, 980)}
    for ri, (y1, y2) in enumerate(rows_y):
        for cname, (x1, x2) in cols.items():
            blocks.append(_ocr(f"{cname}{ri}", x1=x1, x2=x2, y1=y1, y2=y2))
    return blocks


class TestDetectColumns:
    def test_single_column_returns_none(self) -> None:
        cols = detect_columns(
            _single_column_blocks(),
            min_gutter_ratio=0.04,
            min_band_occupancy_ratio=0.10,
        )
        assert cols is None

    def test_three_panels_detected(self) -> None:
        cols = detect_columns(
            _three_panel_blocks(),
            min_gutter_ratio=0.04,
            min_band_occupancy_ratio=0.10,
        )
        assert cols is not None
        assert len(cols) == 3
        # Bands are ordered left-to-right and non-overlapping.
        assert cols[0][0] < cols[1][0] < cols[2][0]
        for (_lo, hi), (lo2, _hi2) in zip(cols, cols[1:]):
            assert hi <= lo2

    def test_too_few_blocks_returns_none(self) -> None:
        blocks = [_ocr("a", x1=0, x2=100, y1=0, y2=20), _ocr("b", x1=0, x2=100, y1=30, y2=50)]
        assert detect_columns(blocks, min_gutter_ratio=0.04, min_band_occupancy_ratio=0.10) is None


class TestStampColumns:
    def test_order_preserved_and_stamped(self) -> None:
        blocks = _three_panel_blocks()  # row-major input
        cols = detect_columns(blocks, min_gutter_ratio=0.04, min_band_occupancy_ratio=0.10)
        assert cols is not None
        stamped = stamp_columns(blocks, cols)
        # Order is unchanged (row-major preserved — tables stay intact).
        assert [b.block_id for b in stamped] == [b.block_id for b in blocks]
        # Every positioned block carries a column_index, and the three columns map
        # to distinct indices L<M<R.
        by_id = {b.block_id: b.extra["column_index"] for b in stamped}
        assert by_id["L0"] < by_id["M0"] < by_id["R0"]
        assert all("column_index" in b.extra for b in stamped)

    def test_full_width_title_is_anchor(self) -> None:
        blocks = _three_panel_blocks()
        cols = detect_columns(blocks, min_gutter_ratio=0.04, min_band_occupancy_ratio=0.10)
        assert cols is not None
        # A full-width title above everything (spans all three columns).
        title = _ocr("TITLE", x1=20, x2=980, y1=-50, y2=-20)
        stamped = stamp_columns([title] + blocks, cols)
        # Title stays first and is NOT assigned a column (neutral anchor).
        assert stamped[0].block_id == "TITLE"
        assert "column_index" not in stamped[0].extra


class TestAnnotateColumns:
    def test_single_column_is_noop(self) -> None:
        blocks = _single_column_blocks()
        result = annotate_columns(blocks, min_gutter_ratio=0.04, min_band_occupancy_ratio=0.10)
        assert [b.block_id for b in result] == [b.block_id for b in blocks]
        assert all("column_index" not in b.extra for b in result)

    def test_multi_panel_stamped_in_place(self) -> None:
        blocks = _three_panel_blocks()
        result = annotate_columns(blocks, min_gutter_ratio=0.04, min_band_occupancy_ratio=0.10)
        # Order preserved; distinct column indices assigned.
        assert [b.block_id for b in result] == [b.block_id for b in blocks]
        assert len({b.extra["column_index"] for b in result}) == 3
