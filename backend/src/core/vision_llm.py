from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from src.core.config import Settings

logger = logging.getLogger(__name__)


_OLLAMA_VISION_MODELS = [
    "qwen2.5-vl",
    "qwen2.5vl",
    "qwen2-vl",
    "minicpm-v",
    "llava",
    "moondream",
    "bakllava",
    "llava-phi3",
]


class VisualVerificationResult(BaseModel):
    supported: bool = True
    confidence: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    unreadable_regions: list[str] = Field(default_factory=list)
    raw_verdict: str = ""


class VisionLLM:
    """Small adapter for VLM generation over retrieved image evidence.

    It mirrors the project's existing local-first posture: try an installed
    Ollama vision model, then fall back to an OpenAI-compatible chat endpoint
    when an API key is configured.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._available_ollama_model: str | None = None
        self._model_checked = False
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.vlm_query_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )

    async def generate_with_images(
        self,
        *,
        prompt: str,
        image_paths: list[Path] | None = None,
        image_bytes: list[bytes] | None = None,
    ) -> str:
        images_b64 = self._encode_inputs(image_paths=image_paths or [], image_bytes=image_bytes or [])
        if not images_b64:
            raise ValueError("VisionLLM.generate_with_images requires at least one image")

        provider = (self.settings.vlm_query_provider or "auto").lower().strip()
        if provider in {"auto", "ollama"}:
            model = await self._detect_ollama_model()
            if model is not None:
                return await self._generate_ollama(model=model, prompt=prompt, images_b64=images_b64)
            if provider == "ollama":
                raise RuntimeError("No local Ollama vision model is available")

        if provider in {"auto", "openai", "openai_compatible"} and self.settings.openai_api_key:
            return await self._generate_openai(prompt=prompt, images_b64=images_b64)

        raise RuntimeError("No VLM provider is available")

    async def verify_with_images(
        self,
        *,
        answer: str,
        prompt_context: str,
        image_paths: list[Path],
    ) -> bool:
        verdict = await self.verify_with_images_structured(
            answer=answer,
            prompt_context=prompt_context,
            image_paths=image_paths,
        )
        return verdict.supported

    async def verify_with_images_structured(
        self,
        *,
        answer: str,
        prompt_context: str,
        image_paths: list[Path],
    ) -> VisualVerificationResult:
        if not image_paths:
            return VisualVerificationResult(supported=True)
        prompt = (
            "Check whether the draft answer is supported by the attached evidence images "
            "and the evidence metadata. Return ONLY valid JSON with keys: "
            "supported (boolean), unsupported_claims (array of strings), "
            "unreadable_regions (array of strings), confidence (number 0 to 1). Mark supported=false for any "
            "claim that is not visible/readable in the cited image evidence.\n\n"
            f"EVIDENCE METADATA:\n{prompt_context[:2500]}\n\n"
            f"DRAFT ANSWER:\n{answer[:1800]}\n\nJSON:"
        )
        raw = await self.generate_with_images(prompt=prompt, image_paths=image_paths)
        try:
            import json
            import re

            text = raw.strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else text)
            return VisualVerificationResult(
                supported=bool(payload.get("supported")),
                confidence=max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
                unsupported_claims=[str(v) for v in payload.get("unsupported_claims") or []],
                unreadable_regions=[str(v) for v in payload.get("unreadable_regions") or []],
                raw_verdict=raw,
            )
        except Exception:
            upper = raw.upper()
            supported = "ALL_SUPPORTED" in upper and "UNSUPPORTED" not in upper.replace("ALL_SUPPORTED", "")
            return VisualVerificationResult(
                supported=supported,
                confidence=0.6 if supported else 0.0,
                unsupported_claims=[] if supported else [raw[:240]],
                unreadable_regions=[],
                raw_verdict=raw,
            )

    async def close(self) -> None:
        await self._client.aclose()

    async def _detect_ollama_model(self) -> str | None:
        if self._model_checked:
            return self._available_ollama_model
        self._model_checked = True
        try:
            response = await self._client.get(f"{self.settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5.0)
            response.raise_for_status()
            installed = {m["name"].split(":")[0]: m["name"] for m in response.json().get("models", [])}
            preferred = self.settings.vlm_query_ollama_model.strip()
            candidates = [preferred] if preferred else []
            candidates.extend(model for model in _OLLAMA_VISION_MODELS if model not in candidates)
            for model in candidates:
                if model in installed:
                    self._available_ollama_model = installed[model]
                    return self._available_ollama_model
        except Exception as exc:
            logger.debug("Ollama vision model detection failed", extra={"error": str(exc)})
        return None

    async def _generate_ollama(self, *, model: str, prompt: str, images_b64: list[str]) -> str:
        response = await self._client.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": images_b64,
                "stream": False,
                "options": {
                    "temperature": self.settings.llm_temperature,
                    "num_predict": self.settings.llm_max_output_tokens,
                    # Visual QA can attach several 1024px figures at once; their
                    # combined vision tokens overflow Ollama's 4096 default and
                    # trigger a 400 exceed_context_size_error without this.
                    "num_ctx": self.settings.vlm_caption_num_ctx,
                },
            },
        )
        response.raise_for_status()
        return str(response.json().get("response") or "").strip()

    async def _generate_openai(self, *, prompt: str, images_b64: list[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
            for image_b64 in images_b64
        )
        response = await self._client.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.vlm_query_openai_model or self.settings.openai_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_output_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()

    def _encode_inputs(self, *, image_paths: list[Path], image_bytes: list[bytes]) -> list[str]:
        encoded: list[str] = []
        for data in image_bytes:
            if data:
                encoded.append(self._encode_image_bytes(data))
        for path in image_paths:
            try:
                if path.exists() and path.stat().st_size > 0:
                    encoded.append(self._encode_image_bytes(path.read_bytes()))
            except Exception as exc:
                logger.debug("Skipping unreadable VLM image", extra={"path": str(path), "error": str(exc)})
        return encoded

    def _encode_image_bytes(self, data: bytes) -> str:
        max_side = max(128, int(self.settings.vlm_query_image_max_side_px))
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return base64.b64encode(data).decode("ascii")
