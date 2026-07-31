from __future__ import annotations

import base64
import logging
import re
import threading
import uuid
from pathlib import Path

from src.core.config import Settings
from src.processing.image_quality_checker import ImageQualityChecker
from src.processing.ocr_engine import EasyOCREngine
from src.processing.types import BlockType, ParsedBlock, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_HANDWRITING_PROMPT_VI = (
    "Bạn là chuyên gia phiên âm tài liệu. Hãy đọc và ghi lại TOÀN BỘ chữ viết tay trong ảnh.\n\n"
    "Quy tắc chung:\n"
    "- Phiên âm chính xác từng ký tự, giữ nguyên cấu trúc xuống dòng, gạch đầu dòng, đánh số.\n"
    "- Tiếng Việt phải đầy đủ dấu thanh (sắc, huyền, hỏi, ngã, nặng) và dấu phụ (ă, â, ê, ô, ơ, ư, đ).\n"
    "- Số, mã, ngày tháng, công thức, ký hiệu toán học/khoa học → sao chép y hệt, không diễn giải.\n"
    "- Hoa thường, gạch chân, dấu ngoặc → giữ nguyên như trong ảnh.\n"
    "- Chữ ký, con dấu → ghi [chữ ký] hoặc [con dấu].\n"
    "- Hình vẽ, sơ đồ, mũi tên minh họa → ghi [hình vẽ] hoặc mô tả ngắn trong ngoặc vuông.\n"
    "- Chữ in sẵn (tiêu đề biểu mẫu, nhãn cột) → BỎ QUA, chỉ lấy phần viết tay.\n"
    "- Chữ không đọc được → ghi [không rõ].\n"
    "- CHỈ trả về văn bản phiên âm, không thêm bình luận, giải thích hay tiêu đề.\n"
)
_HANDWRITING_PROMPT_EN = (
    "You are a document transcription expert. Transcribe ALL handwritten text in this image.\n\n"
    "General rules:\n"
    "- Preserve every character exactly; keep line breaks, bullet points, and numbering as written.\n"
    "- Numbers, codes, dates, formulas, and scientific/mathematical notation → copy verbatim, do not interpret.\n"
    "- Capitalization, underlines, and brackets → preserve as in the image.\n"
    "- Signatures, stamps → write [signature] or [stamp].\n"
    "- Diagrams, drawings, arrows → write [diagram] or a brief bracketed description.\n"
    "- Pre-printed text (form headers, column labels) → IGNORE, transcribe only handwritten parts.\n"
    "- Unclear words → write [unclear].\n"
    "- Return ONLY the transcribed text, no commentary, explanation, or headers.\n"
)

# Vision models tried in priority order — qwen2.5vl reads dense text + Vietnamese far better.
_OLLAMA_VISION_MODELS = ["qwen2.5-vl", "qwen2.5vl", "qwen2-vl", "minicpm-v", "llava", "moondream", "bakllava", "llava-phi3"]


def _remove_repetition(text: str) -> str:
    pattern = re.compile(r"(.{8,}?)\1{3,}", re.DOTALL)
    m = pattern.search(text)
    if m:
        text = text[: m.start()].rstrip(" /,\n") + "..."
    return text


class HandwritingReader:
    def __init__(
        self,
        *,
        settings: Settings,
        quality_checker: ImageQualityChecker | None = None,
        ocr_engine: EasyOCREngine | None = None,
    ) -> None:
        self.settings = settings
        self.quality_checker = quality_checker or ImageQualityChecker(settings)
        self.ocr_engine = ocr_engine or EasyOCREngine(lang="vi", gpu=settings.ocr_easyocr_gpu)
        self._vlm_model: str | None = None
        self._vlm_model_checked: bool = False
        self._vlm_lock = threading.Lock()

    def parse_image(self, image_path: Path, *, language: str = "vi") -> ParsedDocument:
        quality = self.quality_checker.check(image_path)

        # VLM is robust to lighting/contrast issues that break EasyOCR — always try it first.
        # Quality gate only blocks EasyOCR fallback, not VLM.
        vlm_text = self._try_vlm(image_path, language=language)

        if not quality.is_acceptable and not vlm_text:
            return ParsedDocument(
                source_path=str(image_path),
                file_type=image_path.suffix.lower().lstrip("."),
                language=language,
                warnings=quality.warnings,
                extra={
                    "parser": "handwriting_reader",
                    "image_quality_score": quality.score,
                    "accepted_as_evidence": False,
                },
            )

        # VLM already ran above — use it if successful
        if vlm_text:
            page = ParsedPage(
                page_number=1,
                blocks=[
                    ParsedBlock(
                        block_id=str(uuid.uuid4()),
                        block_index=0,
                        block_type=BlockType.HANDWRITING.value,
                        content=vlm_text,
                        page_number=1,
                        language=language,
                        reading_order=0,
                        source="handwriting_reader_vlm",
                        extra={"image_quality_score": quality.score},
                    )
                ],
            )
            return ParsedDocument(
                source_path=str(image_path),
                file_type=image_path.suffix.lower().lstrip("."),
                language=language,
                pages=[page],
                extra={
                    "parser": "handwriting_reader",
                    "image_quality_score": quality.score,
                    "handwriting_source": "vlm",
                    "accepted_as_evidence": True,
                },
            )

        # Fall back to EasyOCR + confidence gate
        parsed = self.ocr_engine.parse_image(image_path, language=language)
        confidences = [block.ocr_confidence for block in parsed.blocks if block.ocr_confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if avg_confidence < self.settings.min_handwriting_confidence:
            parsed.warnings.append("handwriting OCR confidence is below the evidence threshold")
            parsed.pages = []
            parsed.extra.update(
                {
                    "parser": "handwriting_reader",
                    "image_quality_score": quality.score,
                    "handwriting_confidence": avg_confidence,
                    "accepted_as_evidence": False,
                }
            )
            return parsed

        for block in parsed.blocks:
            block.block_type = BlockType.HANDWRITING.value
            block.source = "handwriting_reader"
            block.extra["image_quality_score"] = quality.score
        parsed.extra.update(
            {
                "parser": "handwriting_reader",
                "image_quality_score": quality.score,
                "handwriting_confidence": avg_confidence,
                "handwriting_source": "easyocr",
                "accepted_as_evidence": True,
            }
        )
        return parsed

    def _try_vlm(self, image_path: Path, *, language: str = "vi") -> str:
        """Transcribe handwriting via local VLM. Returns empty string on failure."""
        model = self._detect_vlm_model()
        if model is None:
            return ""
        try:
            import httpx
        except ImportError:
            return ""

        prompt = _HANDWRITING_PROMPT_VI if language == "vi" else _HANDWRITING_PROMPT_EN
        try:
            from src.processing.figure_captioner import FigureCaptioner
            image_b64 = FigureCaptioner._encode_image_resized(image_path, max_side=1024)
        except Exception:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 1024, "num_ctx": 8192},
                },
                timeout=180.0,
            )
            response.raise_for_status()
            text = _remove_repetition((response.json().get("response") or "").strip())
            if len(text.strip()) >= 5:
                logger.info(
                    "Handwriting transcribed via VLM",
                    extra={"model": model, "image": image_path.name, "chars": len(text)},
                )
                return text
        except Exception as exc:
            logger.warning(
                "VLM handwriting transcription failed",
                extra={"model": model, "error": str(exc), "error_type": type(exc).__name__},
            )
        return ""

    def _detect_vlm_model(self) -> str | None:
        with self._vlm_lock:
            if self._vlm_model_checked:
                return self._vlm_model
            self._vlm_model_checked = True
            try:
                import httpx
                resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=5.0)
                resp.raise_for_status()
                installed = {m["name"].split(":")[0]: m["name"] for m in resp.json().get("models", [])}
                for model in _OLLAMA_VISION_MODELS:
                    if model in installed:
                        self._vlm_model = installed[model]
                        logger.info("Handwriting reader using VLM: %s", self._vlm_model)
                        return self._vlm_model
            except Exception as exc:
                logger.debug("Ollama not reachable for handwriting VLM: %s", exc)
            return None
