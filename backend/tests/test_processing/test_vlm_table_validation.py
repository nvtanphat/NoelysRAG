from __future__ import annotations

from src.services.parse_index_pipeline import _table_has_numeric_data

_RATIO = 0.15


class TestTableHasNumericData:
    def test_dataless_table_rejected(self) -> None:
        # The exact failure mode: header-only grid, all body cells empty (no digits).
        caption = (
            "### Tóm tắt thông tin tài chính 5 năm gần nhất\n"
            "| **Mã hàng** | **Năm** |\n"
            "|--------------|----------|\n"
            "| 2020 | |\n"
            "| 2021 | |\n"
            "| 2022 | |\n"
            "| 2023 | |\n"
        )
        # The only digits sit in the FIRST column (row headers); the data columns
        # are empty. Ratio of digit-bearing body cells is well under 0.15.
        assert _table_has_numeric_data(caption, min_digit_cell_ratio=_RATIO) is False

    def test_real_financial_table_accepted(self) -> None:
        caption = (
            "| Chỉ tiêu | 2020 | 2021 | 2022 |\n"
            "| --- | --- | --- | --- |\n"
            "| Lợi nhuận trước thuế | 13519 | 12922 | 10968 |\n"
            "| Lợi nhuận sau thuế | 11236 | 10633 | 8578 |\n"
        )
        assert _table_has_numeric_data(caption, min_digit_cell_ratio=_RATIO) is True

    def test_prose_caption_accepted(self) -> None:
        # No markdown table parses → not the dataless-table bug → accept.
        caption = (
            "Biểu đồ cho thấy lợi nhuận của công ty tăng đều qua các năm, "
            "phản ánh hiệu quả hoạt động kinh doanh ổn định trong giai đoạn."
        )
        assert _table_has_numeric_data(caption, min_digit_cell_ratio=_RATIO) is True

    def test_multi_section_pools_cells(self) -> None:
        # One empty section + one real section: pooled digit ratio clears the bar.
        caption = (
            "## Bảng rỗng\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| | |\n"
            "## Bảng số\n"
            "| Năm | Giá trị |\n"
            "| --- | --- |\n"
            "| 2020 | 13519 |\n"
            "| 2021 | 12922 |\n"
        )
        assert _table_has_numeric_data(caption, min_digit_cell_ratio=_RATIO) is True
