"""VLM-based figure captioner for charts, diagrams, and images in documents.

Flow:
  1. Receive a figure image (path or cropped region bytes).
  2. Try a local vision-capable Ollama model (qwen2-vl or llava).
  3. Fall back to PaddleOCR if VLM is unavailable / returns empty.
  4. Return a plain-text caption to be stored as the FIGURE block content.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import threading
from pathlib import Path

from src.processing.types import BBox

logger = logging.getLogger(__name__)

# VLM prompt — engineered for stylized/decorative Vietnamese text.
# Key tactics:
#   - Force structured Markdown output (sections + bullets) — easier for chunking
#   - Explicit Vietnamese accent handling — model often drops diacritics
#   - Reject hallucination ("if unsure, write [?]")
#   - List terms separately to avoid run-on gibberish
_CAPTION_PROMPT_VI = (
    "Bạn là chuyên gia trích xuất nội dung tài liệu. Đọc hình ảnh và xuất ra DỮ LIỆU THỰC — không mô tả hình thức.\n"
    "\n"
    "Quan sát toàn bộ ảnh, xác định từng vùng nội dung, rồi áp dụng quy tắc phù hợp:\n"
    "\n"
    "- **Bảng:** Xuất Markdown table giữ nguyên hàng × cột. Ghi rõ đơn vị nếu có. Không bỏ sót hàng.\n"
    "- **Biểu đồ cột/đường/vùng:** Xuất Markdown table theo 3 bước:\n"
    "  Bước 1 – Đọc chú giải (legend): liệt kê từng chuỗi dữ liệu theo thứ tự xuất hiện trong legend. "
    "Dùng ĐÚNG nhãn chú giải làm cột đầu tiên (row header) — ví dụ 'Lợi nhuận trước thuế', 'Lợi nhuận sau thuế'. "
    "Không tự đặt tên 'Chuỗi 1' hay 'Cột 2'.\n"
    "  Bước 2 – Dòng HEADER: đọc nhãn trục X từ TRÁI qua PHẢI, từng nhãn nằm dưới mỗi nhóm cột (ví dụ 2019, 2020, 2021, 2022, 2023). "
    "KHÔNG lấy khoảng năm từ tiêu đề biểu đồ — tiêu đề có thể ghi '2019–2023' nhưng phải đọc từng nhãn riêng trên trục X.\n"
    "  Bước 3 – Điền giá trị: với mỗi chuỗi, đọc số in trên đỉnh cột, khớp đúng với nhãn năm bên dưới cột đó. "
    "ĐẾM số nhóm cột = số nhãn trục X rồi điền ĐỦ bấy nhiêu giá trị — không bỏ sót, không gộp. Không đọc được → ghi [không rõ]. Không mô tả xu hướng.\n"
    "- **Biểu đồ MỘT chuỗi (không có legend):** tên chỉ tiêu nằm ở TIÊU ĐỀ ngay trên biểu đồ (ví dụ 'Lợi nhuận sau thuế (tỷ đồng)') — dùng làm tên cột giá trị. "
    "Xuất bảng dọc: `| Năm | <tên chỉ tiêu> |`, mỗi năm một hàng. KHÔNG tự đặt 'Giá trị' hay 'Cột 2' nếu đã có tên chỉ tiêu ở tiêu đề.\n"
    "- **Trang NHIỀU biểu đồ nhỏ (dashboard):** xử lý TỪNG biểu đồ RIÊNG. Mỗi biểu đồ mở đầu bằng `## <tên chỉ tiêu lấy từ tiêu đề biểu đồ>` rồi tới bảng của riêng nó. "
    "TUYỆT ĐỐI không trộn số liệu của các biểu đồ khác nhau vào cùng một bảng, không để một biểu đồ thiếu tên chỉ tiêu.\n"
    "- **Biểu đồ tròn/donut:** Liệt kê từng phần: tên — giá trị hoặc %.\n"
    "- **Sơ đồ phân cấp/tổ chức:** Dùng indent thể hiện quan hệ cha–con.\n"
    "- **Lưu đồ/quy trình:** Liệt kê các bước theo thứ tự, giữ nguyên nhãn mũi tên/điều kiện.\n"
    "- **Văn bản/tiêu đề/chú thích:** Ghi nguyên văn, đủ dấu tiếng Việt.\n"
    "- **Công thức/ký hiệu:** Sao chép y hệt, không diễn giải.\n"
    "- **Ảnh thực/minh họa (không có dữ liệu cấu trúc):** Mô tả ngắn gọn: chủ thể chính là gì/ai, bối cảnh, hành động nếu có. Tối đa 3 câu.\n"
    "\n"
    "Ví dụ trang nhiều biểu đồ nhỏ (mỗi biểu đồ một vùng riêng, lấy tên chỉ tiêu từ tiêu đề):\n"
    "## Doanh thu thuần (tỷ đồng)\n"
    "| Năm | Doanh thu thuần (tỷ đồng) |\n"
    "| --- | --- |\n"
    "| 2023 | 60.369 |\n"
    "| 2024 | 61.783 |\n"
    "## Lợi nhuận sau thuế (tỷ đồng)\n"
    "| Năm | Lợi nhuận sau thuế (tỷ đồng) |\n"
    "| --- | --- |\n"
    "| 2023 | 9.019 |\n"
    "| 2024 | 9.453 |\n"
    "\n"
    "Quy tắc bắt buộc:\n"
    "- Tiếng Việt phải đủ dấu thanh và dấu phụ.\n"
    "- KHÔNG mô tả màu sắc, kích thước, vị trí trừ khi đó là thông tin duy nhất.\n"
    "- KHÔNG suy diễn. Không đọc được → ghi [không rõ].\n"
    "- Nhiều vùng nội dung → dùng ## tên vùng làm header phân tách.\n"
    "- CHỈ trả về nội dung trích xuất, không thêm lời dẫn.\n"
)
_CAPTION_PROMPT_EN = (
    "You are a document content extraction expert. Read the image and output the ACTUAL DATA — do not describe appearance.\n"
    "\n"
    "Survey the whole image, identify each content region, then apply the matching rule:\n"
    "\n"
    "- **Table:** Output a Markdown table preserving all rows × columns. Include units if shown. Do not skip rows.\n"
    "- **Bar/line/area chart:** Output a Markdown table in 3 steps:\n"
    "  Step 1 – Read the legend: list each data series in the order they appear in the legend. "
    "Use the EXACT legend label as the first column (row header) — e.g. 'Pre-tax profit', 'Post-tax profit'. "
    "Do not invent names like 'Series 1' or 'Column 2'.\n"
    "  Step 2 – Header row: read X-axis labels LEFT to RIGHT, one label per bar group (e.g. 2019, 2020, 2021, 2022, 2023). "
    "Read the tick label printed below each bar group — do NOT use the year range from the chart title (e.g. a title saying '2019–2023' is not the same as reading individual axis labels).\n"
    "  Step 3 – Fill values: for each series, read the number printed on top of its bar, matched to the year label directly below that bar. "
    "COUNT bar groups = number of X-axis labels; fill EXACTLY that many values per row — do not skip or merge. Unreadable → write [unclear]. Do not describe trends.\n"
    "- **SINGLE-series chart (no legend):** the metric name is the TITLE right above the chart (e.g. 'Post-tax profit') — use it as the value column name. "
    "Output a vertical table: `| Year | <metric name> |`, one row per year. Do not invent 'Value' or 'Column 2' when the title already names the metric.\n"
    "- **Page with MANY small charts (dashboard):** handle EACH chart SEPARATELY. Start each chart with `## <metric name from its title>` followed by its own table. "
    "NEVER mix values from different charts into one table, and never leave a chart without its metric name.\n"
    "- **Pie/donut chart:** List each segment: label — value or percentage.\n"
    "- **Hierarchy/org chart:** Use indentation to show parent–child relationships.\n"
    "- **Flowchart/process:** List steps in order, preserving arrow labels and conditions.\n"
    "- **Text/titles/captions:** Transcribe verbatim.\n"
    "- **Formulas/symbols:** Copy exactly, do not interpret.\n"
    "- **Natural image/illustration (no structured data):** Write a brief factual caption: main subject, setting, action if any. Max 3 sentences.\n"
    "\n"
    "Example of a multi-chart page (one region per chart, metric name taken from each title):\n"
    "## Net revenue (VND bn)\n"
    "| Year | Net revenue (VND bn) |\n"
    "| --- | --- |\n"
    "| 2023 | 60.369 |\n"
    "| 2024 | 61.783 |\n"
    "## Post-tax profit (VND bn)\n"
    "| Year | Post-tax profit (VND bn) |\n"
    "| --- | --- |\n"
    "| 2023 | 9.019 |\n"
    "| 2024 | 9.453 |\n"
    "\n"
    "Universal rules:\n"
    "- Do NOT describe colors, sizes, or positions unless that is the only meaningful content.\n"
    "- Do NOT infer. If unreadable → write [unclear].\n"
    "- Multiple content regions → use ## region-title as a separator header.\n"
    "- Return ONLY the extracted content, no preamble.\n"
)


_BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s*(\*{0,2})(.+?)(\*{0,2})\s*$")


def _looks_like_list_hallucination(
    text: str,
    *,
    bullet_ratio_max: float = 0.70,
    bullet_max_words: int = 3,
) -> bool:
    """Detect VLM captions that are just lists of short items (e.g. ImageNet animal names).

    When a decorative photo slide is captioned, VLMs often enumerate objects
    ("- Quail", "- otter", "- grouse") instead of describing the image. These
    captions are semantically useless and pollute retrieval.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    short_bullet_count = 0
    for line in lines:
        m = _BULLET_LINE_RE.match(line)
        if m:
            item_text = m.group(2).strip()
            if len(item_text.split()) <= bullet_max_words:
                short_bullet_count += 1
    return (short_bullet_count / len(lines)) > bullet_ratio_max


def _looks_like_gibberish(text: str) -> bool:
    """Detect VLM hallucination / garbled output:
       - Long runs of letters without spaces (>15 chars, no space)
       - Mixed scripts mid-word
       - Very low space-to-char ratio
    """
    if not text or len(text) < 30:
        return False
    # Check for word-monster: 15+ letters without space
    long_runs = re.findall(r"[a-zA-ZÀ-ỹ]{15,}", text)
    if len(long_runs) >= 2:
        return True
    # Check space ratio for non-Asian text
    if re.search(r"[a-zA-Z]", text):
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len > 12:  # words too long on average
            return True
    return False

# Vision models supported by Ollama, tried in order of document/table strength.
# qwen2.5vl / qwen2-vl read dense tables + Vietnamese far better than minicpm-v,
# which hallucinates on number-dense financial pages. minicpm-v kept as fallback.
_OLLAMA_VISION_MODELS = ["qwen2.5-vl", "qwen2.5vl", "qwen2-vl", "minicpm-v", "llava", "moondream", "bakllava", "llava-phi3"]


class FigureCaptioner:
    """Caption document figures using a local VLM with OCR fallback."""

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        language: str = "vi",
        ocr_fallback: bool = True,
        timeout: float = 300.0,
        image_max_side_px: int = 1024,
        num_ctx: int = 8192,
        num_predict: int = 512,
        list_bullet_ratio_max: float = 0.70,
        list_bullet_max_words: int = 3,
        ollama_model: str | None = None,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.language = language
        self.ocr_fallback = ocr_fallback
        self.timeout = timeout
        self.image_max_side_px = image_max_side_px
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.list_bullet_ratio_max = list_bullet_ratio_max
        self.list_bullet_max_words = list_bullet_max_words
        self.ollama_model = ollama_model
        self._available_model: str | None = None
        self._model_checked: bool = False
        self._ocr_engine = None  # lazy-init, reused across all figures
        self._ocr_engine_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def caption_image_path(self, image_path: Path) -> str:
        """Return a caption for a standalone image file."""
        return self._caption(image_path=image_path)

    def caption_page_region(
        self,
        page_image_path: Path,
        bbox: BBox,
        *,
        page_width: int,
        page_height: int,
    ) -> str:
        """Crop a bbox region from a page image and return a caption.

        bbox coordinates are in Docling's page coordinate system (points).
        """
        caption, _ = self.caption_page_region_with_path(
            page_image_path, bbox, page_width=page_width, page_height=page_height
        )
        return caption

    def caption_page_region_with_path(
        self,
        page_image_path: Path,
        bbox: BBox,
        *,
        page_width: int,
        page_height: int,
    ) -> tuple[str, Path | None]:
        """Same as caption_page_region but also returns the crop file path.

        The crop path is needed by the visual embedding pipeline to read the
        image bytes for SigLIP. Returns (caption, crop_path | None).
        """
        cropped = self._crop_region(page_image_path, bbox, page_width=page_width, page_height=page_height)
        if cropped is None:
            return "", None
        return self._caption(image_path=cropped), cropped

    def unload(self) -> None:
        """Release lazy-loaded resources (OCR engine) to free RAM.

        Safe to call multiple times. Should be called after all figures in a
        document have been captioned, before the visual embedding step starts.
        """
        import gc
        with self._ocr_engine_lock:
            if self._ocr_engine is not None:
                self._ocr_engine = None
        gc.collect()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _caption(self, *, image_path: Path) -> str:
        if not image_path.exists() or image_path.stat().st_size < 512:
            return ""

        # Try VLM first
        vlm_caption = self._try_vlm(image_path)
        if vlm_caption:
            return vlm_caption

        # Fall back to OCR (extracts any text that appears in the figure)
        if self.ocr_fallback:
            return self._try_ocr_fallback(image_path)

        return ""

    def _try_vlm(self, image_path: Path) -> str:
        model = self._detect_available_model()
        if model is None:
            return ""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — skipping VLM captioning")
            return ""

        prompt = _CAPTION_PROMPT_VI if self.language == "vi" else _CAPTION_PROMPT_EN
        image_b64 = self._encode_image_resized(image_path, max_side=self.image_max_side_px)

        try:
            response = httpx.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": self.num_predict, "num_ctx": self.num_ctx},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            caption = (data.get("response") or "").strip()
            caption = self._strip_special_tokens(caption)
            caption = self._remove_repetition_loops(caption)
            if self._looks_like_cross_language_hallucination(caption):
                logger.warning(
                    "VLM caption rejected due to cross-language hallucination",
                    extra={"model": model, "image": image_path.name, "chars": len(caption)},
                )
                return ""
            if _looks_like_list_hallucination(
                caption,
                bullet_ratio_max=self.list_bullet_ratio_max,
                bullet_max_words=self.list_bullet_max_words,
            ):
                logger.warning(
                    "VLM caption rejected: list-only hallucination (decorative image)",
                    extra={"model": model, "image": image_path.name, "chars": len(caption)},
                )
                return ""
            if caption:
                logger.info(
                    "Figure captioned via VLM",
                    extra={"model": model, "image": image_path.name, "chars": len(caption)},
                )
            return caption
        except Exception as exc:
            body = ""
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = resp.text[:500]
                except Exception:
                    body = ""
            logger.warning(
                "VLM captioning failed",
                extra={"model": model, "error": str(exc), "error_type": type(exc).__name__, "body": body},
            )
            return ""

    def _looks_like_cross_language_hallucination(self, text: str) -> bool:
        """Reject VLM captions that inject unrelated CJK text into vi/en OCR output."""
        if not text or self.language not in {"vi", "en"}:
            return False

        cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        if not cjk_chars:
            return False

        visible_chars = [ch for ch in text if not ch.isspace()]
        cjk_ratio = len(cjk_chars) / max(1, len(visible_chars))

        # A genuine Vietnamese/English slide should not contain sustained CJK runs.
        # Short isolated symbols are tolerated, but VLM hallucinations often contain
        # repeated mixed fragments such as "đ用力導決策".
        has_cjk_run = re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}", text) is not None
        has_mixed_garble = re.search(r"[A-Za-zÀ-ỹ][\u3400-\u4dbf\u4e00-\u9fff]|[\u3400-\u4dbf\u4e00-\u9fff][A-Za-zÀ-ỹ]", text) is not None
        return cjk_ratio >= 0.02 or has_cjk_run or has_mixed_garble

    def _detect_available_model(self) -> str | None:
        if self._model_checked:
            return self._available_model
        self._model_checked = True
        try:
            import httpx
            resp = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            all_models = resp.json().get("models", [])
            installed = {m["name"].split(":")[0]: m["name"] for m in all_models}
            
            if self.ollama_model:
                pref = self.ollama_model.strip()
                if pref in [m["name"] for m in all_models]:
                    self._available_model = pref
                    logger.info("Figure captioner using configured VLM model: %s", self._available_model)
                    return self._available_model
                pref_base = pref.split(":")[0]
                if pref_base in installed:
                    self._available_model = installed[pref_base]
                    logger.info("Figure captioner using configured VLM model: %s", self._available_model)
                    return self._available_model

            for model in _OLLAMA_VISION_MODELS:
                if model in installed:
                    self._available_model = installed[model]
                    logger.info("Figure captioner using VLM model: %s", self._available_model)
                    return self._available_model
        except Exception as exc:
            logger.warning("Ollama not reachable for figure captioning: %s", exc)
        logger.info("No vision model found in Ollama — figure captioner will use OCR fallback")
        return None

    @staticmethod
    def _encode_image_resized(image_path: Path, max_side: int = 1024) -> str:
        """Resize image to max_side px on longest side, return base64. Falls back to raw bytes."""
        try:
            from PIL import Image
            import io
            img = Image.open(image_path)
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return base64.b64encode(image_path.read_bytes()).decode("ascii")

    @staticmethod
    def _strip_special_tokens(text: str) -> str:
        """Remove model special tokens that leak into captions of token-labelled
        figures (e.g. attention-visualization diagrams print ``<pad>`` / ``<EOS>``
        on the image, and the VLM transcribes them verbatim — pure noise for
        retrieval).
        """
        if not text:
            return text
        text = re.sub(r"<\s*/?\s*(pad|eos|bos|s|unk|sep|cls|mask)\s*>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<\|[^>]*\|>", " ", text)  # chat-template markers like <|im_end|>
        # Collapse the whitespace runs left behind, per line to keep table rows.
        text = "\n".join(" ".join(line.split()) for line in text.splitlines())
        return text.strip()

    @staticmethod
    def _remove_repetition_loops(text: str) -> str:
        """Detect and truncate repeating token loops from VLM hallucination."""
        if not text:
            return text
        # Find repeating substrings >= 8 chars that appear 4+ times consecutively
        import re
        pattern = re.compile(r"(.{8,}?)\1{3,}", re.DOTALL)
        m = pattern.search(text)
        if m:
            text = text[:m.start()].rstrip(" /,\n") + "..."
        return text

    def _try_ocr_fallback(self, image_path: Path) -> str:
        """Run EasyOCR on the figure region to extract any embedded text."""
        try:
            if self._ocr_engine is None:
                with self._ocr_engine_lock:
                    if self._ocr_engine is None:
                        from src.processing.ocr_engine import EasyOCREngine
                        from src.core.config import get_settings as _get_settings
                        self._ocr_engine = EasyOCREngine(lang="vi" if self.language == "vi" else "en", gpu=_get_settings().ocr_easyocr_gpu)
            parsed = self._ocr_engine.parse_image(image_path, language=self.language)
            text = " ".join(b.content for b in parsed.blocks if b.content.strip())
            if text:
                prefix = "[Hình - văn bản trích xuất]" if self.language == "vi" else "[Figure - extracted text]"
                return f"{prefix}: {text}"
        except Exception as exc:
            logger.debug("OCR fallback for figure failed: %s", exc)
        return ""

    @staticmethod
    def _crop_region(
        page_image_path: Path,
        bbox: BBox,
        *,
        page_width: int,
        page_height: int,
    ) -> Path | None:
        """Crop a region from a page image using Docling bbox (point coordinates)."""
        try:
            import cv2
        except ImportError:
            logger.debug("cv2 not available — cannot crop figure region")
            return None

        img = cv2.imread(str(page_image_path))
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        pw = max(1, page_width)
        ph = max(1, page_height)

        # Docling bbox: l/t/r/b in page points (origin = bottom-left).
        # Convert to pixel coordinates (origin = top-left).
        px1 = int(bbox.x1 / pw * img_w)
        px2 = int(bbox.x2 / pw * img_w)
        py1 = int((ph - bbox.y2) / ph * img_h)  # flip y-axis
        py2 = int((ph - bbox.y1) / ph * img_h)

        # Clamp to image bounds
        px1, px2 = max(0, px1), min(img_w, px2)
        py1, py2 = max(0, py1), min(img_h, py2)

        if px2 - px1 < 10 or py2 - py1 < 10:
            return None

        cropped = img[py1:py2, px1:px2]
        digest = hashlib.sha1(f"{page_image_path}:{bbox.x1:.1f},{bbox.y1:.1f}".encode()).hexdigest()[:12]
        out_dir = page_image_path.parent / "figure_crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fig-{digest}.png"
        cv2.imwrite(str(out_path), cropped)
        return out_path
