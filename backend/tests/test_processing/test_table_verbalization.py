from __future__ import annotations

from src.processing.table_serializer import verbalize_rows


class TestSectionedVerbalization:
    def test_sectioned_keyvalue_table_keeps_metric_labels(self) -> None:
        # The shape a small VLM emits for a multi-metric chart page: a 2-column
        # key/value table where each metric is a section-header row (only col 0
        # filled) followed by year|value rows.
        header = ["Tổng tài sản (tỷ đồng)", ""]
        rows = [
            ["2020", "48.435"],
            ["2021", "53.332"],
            ["Lợi nhuận trước thuế (tỷ đồng)", ""],   # section header
            ["2022", "10.496"],
            ["2024", "11.600"],
            ["Lợi nhuận sau thuế (tỷ đồng)", ""],     # section header
            ["2022", "8.578"],
            ["2024", "9.453"],
        ]
        out = verbalize_rows(header, rows, table_name="vinamilk")
        text = "\n".join(s for _, s in out)

        # Each figure stays tied to its real metric — NOT to "Tổng tài sản".
        assert "Lợi nhuận trước thuế (tỷ đồng). 2022: 10.496" in text
        assert "Lợi nhuận trước thuế (tỷ đồng). 2024: 11.600" in text
        assert "Lợi nhuận sau thuế (tỷ đồng). 2022: 8.578" in text
        assert "Lợi nhuận sau thuế (tỷ đồng). 2024: 9.453" in text
        # The first metric (carried from the header) is correct too.
        assert "Tổng tài sản (tỷ đồng). 2020: 48.435" in text
        # No data row is mislabeled as a different metric.
        assert "Tổng tài sản (tỷ đồng). 2022: 10.496" not in text
        # Section-header rows themselves are not emitted as data rows.
        assert ": Lợi nhuận trước thuế (tỷ đồng)." not in text

    def test_normal_wide_table_unchanged(self) -> None:
        # A normal grid (every data row has >= 2 filled cells) must use the
        # existing per-column verbalization, untouched by the section logic.
        header = ["Chỉ tiêu", "2020", "2021"]
        rows = [
            ["Lợi nhuận trước thuế", "13.519", "12.922"],
            ["Lợi nhuận sau thuế", "11.236", "10.633"],
        ]
        out = verbalize_rows(header, rows, table_name="t")
        text = "\n".join(s for _, s in out)
        assert "Chỉ tiêu: Lợi nhuận trước thuế. 2020: 13.519. 2021: 12.922" in text
        assert "Chỉ tiêu: Lợi nhuận sau thuế. 2020: 11.236. 2021: 10.633" in text

    def test_english_sectioned(self) -> None:
        header = ["Total assets", ""]
        rows = [
            ["2020", "48435"],
            ["Pre-tax profit", ""],
            ["2022", "10496"],
        ]
        out = verbalize_rows(header, rows, table_name="t", language="en")
        text = "\n".join(s for _, s in out)
        assert "Row" in text and "Pre-tax profit. 2022: 10496" in text
