from __future__ import annotations

import asyncio
import logging
import os
import threading

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

from src.core.config import Settings
from src.processing.types import DependencyUnavailableError
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Process-wide cache for the cross-encoder model. Without this, every
# CrossEncoderReranker instance (query_service, summary_service, study_guide,
# inference_engine, the startup warmup, and SLEC sentence-coverage) loads its OWN
# ~2GB cross-encoder — the warmup's copy is even discarded, so the first real
# query reloads it cold and re-inits it again, costing minutes. Keyed by
# (model, device, max_length) so all instances share ONE GPU-resident model.
_CROSS_ENCODER_CACHE: dict[tuple[str, str, int], object] = {}
_CROSS_ENCODER_CACHE_LOCK = threading.Lock()


def get_cached_cross_encoder(model_name: str, device: str, max_length: int):
    key = (model_name, device, max_length)
    cached = _CROSS_ENCODER_CACHE.get(key)
    if cached is not None:
        return cached
    with _CROSS_ENCODER_CACHE_LOCK:
        cached = _CROSS_ENCODER_CACHE.get(key)
        if cached is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                raise DependencyUnavailableError(
                    f"sentence-transformers could not be imported: {exc}"
                ) from exc
            logger.info(
                "Loading CrossEncoder reranker (cached)",
                extra={"model": model_name, "device": device, "max_length": max_length},
            )
            # max_length truncates each (query, chunk) pair so a long chunk (e.g. a
            # 10k-char table) does not blow a predict batch up to minutes.
            cached = CrossEncoder(model_name, device=device, max_length=max_length)
            _CROSS_ENCODER_CACHE[key] = cached
        return cached


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._model_load_lock = asyncio.Lock()
        self._predict_semaphore = asyncio.Semaphore(1)

    @property
    def model(self):
        return self._load_model()

    def _load_model(self):
        if self._model is None:
            self._model = get_cached_cross_encoder(
                self.settings.reranker_model_name,
                self.settings.reranker_device,
                getattr(self.settings, "reranker_max_length", 512),
            )
        return self._model

    async def _aload_model(self):
        if self._model is not None:
            return self._model
        async with self._model_load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model)
            return self._model

    def rerank(self, *, query: str, chunks: list[RetrievedChunk], limit: int | None = None) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        if not self.settings.reranker_enabled:
            return self._fallback(candidates, limit)
        try:
            text_candidates = [chunk for chunk in candidates if not self._is_visual_chunk(chunk)]
            pairs = [(query, chunk.content) for chunk in text_candidates]
            scores = self.model.predict(pairs) if pairs else []
            scored = self._merge_scores_preserving_visual(candidates, text_candidates, scores)
            scored.sort(key=self._rank_key, reverse=True)
            return scored[: limit or self.settings.final_top_k]
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed, falling back to fused scores",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return self._fallback(candidates, limit)

    async def arerank(self, *, query: str, chunks: list[RetrievedChunk], limit: int | None = None) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        if not self.settings.reranker_enabled:
            return self._fallback(candidates, limit)
        try:
            text_candidates = [chunk for chunk in candidates if not self._is_visual_chunk(chunk)]
            pairs = [(query, chunk.content) for chunk in text_candidates]
            model = await self._aload_model()
            if pairs:
                async with self._predict_semaphore:
                    scores = await asyncio.to_thread(model.predict, pairs)
            else:
                scores = []
            scored = self._merge_scores_preserving_visual(candidates, text_candidates, scores)
            scored.sort(key=self._rank_key, reverse=True)
            return scored[: limit or self.settings.final_top_k]
        except Exception as exc:
            logger.warning(
                "CrossEncoder async reranking failed, falling back to fused scores",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return self._fallback(candidates, limit)

    def rerank_multilingual(
        self,
        *,
        queries: list[str],
        chunks: list[RetrievedChunk],
        limit: int | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        unique_queries = list(dict.fromkeys(q for q in queries if q.strip()))
        if not unique_queries:
            unique_queries = [""]

        # Cap number of query variants to stay within reranker_max_pairs budget.
        max_queries = max(1, self.settings.reranker_max_pairs // max(1, len(candidates)))
        if len(unique_queries) > max_queries:
            logger.info(
                "Truncating reranker queries to stay under pair cap",
                extra={"query_count": len(unique_queries), "max_queries": max_queries, "candidate_count": len(candidates)},
            )
            unique_queries = unique_queries[:max_queries]

        if not self.settings.reranker_enabled:
            scored = self._fallback(candidates, limit=None)
        else:
            try:
                text_candidates = [chunk for chunk in candidates if not self._is_visual_chunk(chunk)]
                pairs = [(query, chunk.content) for chunk in text_candidates for query in unique_queries]
                scores = [float(s) for s in self.model.predict(pairs)] if pairs else []
                query_count = len(unique_queries)
                best_scores: dict[str, float] = {}
                for index, chunk in enumerate(text_candidates):
                    start = index * query_count
                    best_scores[chunk.chunk_id] = max(scores[start: start + query_count])
                scored = [
                    chunk.model_copy(
                        update={"rerank_score": self._visual_rank_score(chunk) if self._is_visual_chunk(chunk) else best_scores.get(chunk.chunk_id, 0.0)}
                    )
                    for chunk in candidates
                ]
                scored.sort(key=self._rank_key, reverse=True)
            except Exception as exc:
                logger.warning(
                    "Multilingual CrossEncoder reranking failed, falling back to fused scores",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
                scored = self._fallback(candidates, limit=None)

        final_limit = limit or self.settings.final_top_k
        if use_mmr and len(scored) > 1:
            return self.apply_mmr(scored, limit=final_limit)
        return scored[:final_limit]

    async def arerank_multilingual(
        self,
        *,
        queries: list[str],
        chunks: list[RetrievedChunk],
        limit: int | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        unique_queries = list(dict.fromkeys(q for q in queries if q.strip()))
        if not unique_queries:
            unique_queries = [""]

        max_queries = max(1, self.settings.reranker_max_pairs // max(1, len(candidates)))
        if len(unique_queries) > max_queries:
            logger.info(
                "Truncating reranker queries to stay under pair cap",
                extra={"query_count": len(unique_queries), "max_queries": max_queries, "candidate_count": len(candidates)},
            )
            unique_queries = unique_queries[:max_queries]

        if not self.settings.reranker_enabled:
            scored = self._fallback(candidates, limit=None)
        else:
            try:
                text_candidates = [chunk for chunk in candidates if not self._is_visual_chunk(chunk)]
                pairs = [(query, chunk.content) for chunk in text_candidates for query in unique_queries]
                model = await self._aload_model()
                if pairs:
                    async with self._predict_semaphore:
                        raw_scores = await asyncio.to_thread(model.predict, pairs)
                else:
                    raw_scores = []
                scores = [float(s) for s in raw_scores]
                query_count = len(unique_queries)
                best_scores: dict[str, float] = {}
                for index, chunk in enumerate(text_candidates):
                    start = index * query_count
                    best_scores[chunk.chunk_id] = max(scores[start: start + query_count])
                scored = [
                    chunk.model_copy(
                        update={"rerank_score": self._visual_rank_score(chunk) if self._is_visual_chunk(chunk) else best_scores.get(chunk.chunk_id, 0.0)}
                    )
                    for chunk in candidates
                ]
                scored.sort(key=self._rank_key, reverse=True)
            except Exception as exc:
                logger.warning(
                    "Async multilingual CrossEncoder reranking failed, falling back to fused scores",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
                scored = self._fallback(candidates, limit=None)

        final_limit = limit or self.settings.final_top_k
        if use_mmr and len(scored) > 1:
            return self.apply_mmr(scored, limit=final_limit)
        return scored[:final_limit]

    def apply_mmr(
        self,
        chunks: list[RetrievedChunk],
        *,
        limit: int | None = None,
        lambda_: float = 0.7,
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance: select top-k chunks that balance relevance with diversity.

        lambda_=1.0 → pure relevance ranking (identical to score sort)
        lambda_=0.0 → pure diversity (greedy maximum coverage)
        Default 0.7 favours relevance while still penalising near-duplicate chunks.
        """
        n = limit or len(chunks)
        if len(chunks) <= 1 or n <= 1:
            return chunks[:n]

        selected: list[RetrievedChunk] = [chunks[0]]
        remaining = list(chunks[1:])

        while remaining and len(selected) < n:
            best_mmr = float("-inf")
            best_chunk = remaining[0]
            for candidate in remaining:
                rel = (
                    candidate.rerank_score
                    if candidate.rerank_score is not None
                    else (candidate.fused_score or 0.0)
                )
                max_sim = max(self._jaccard(candidate.content, sel.content) for sel in selected)
                mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_chunk = candidate
            selected.append(best_chunk)
            remaining.remove(best_chunk)

        return selected

    def _fallback(self, chunks: list[RetrievedChunk], limit: int | None) -> list[RetrievedChunk]:
        fallback = sorted(chunks, key=lambda c: c.fused_score, reverse=True)
        return fallback[: limit or self.settings.final_top_k]

    @staticmethod
    def _is_visual_chunk(chunk: RetrievedChunk) -> bool:
        modality = (chunk.modality or "").lower()
        kind = str((chunk.metadata or {}).get("kind") or (chunk.metadata or {}).get("evidence_kind") or "").lower()
        return modality in {"figure", "image", "visual"} or kind == "visual"

    @staticmethod
    def _visual_rank_score(chunk: RetrievedChunk) -> float:
        return float(chunk.fused_score or chunk.graph_score or 0.0)

    def _merge_scores_preserving_visual(
        self,
        candidates: list[RetrievedChunk],
        text_candidates: list[RetrievedChunk],
        scores,
    ) -> list[RetrievedChunk]:
        score_by_id = {
            chunk.chunk_id: float(score)
            for chunk, score in zip(text_candidates, scores, strict=True)
        }
        penalty = getattr(self.settings, "vlm_list_caption_score_penalty", 0.40)
        result = []
        for chunk in candidates:
            if self._is_visual_chunk(chunk):
                raw = self._visual_rank_score(chunk)
            else:
                raw = score_by_id.get(chunk.chunk_id, 0.0)
            if penalty > 0 and self._is_list_caption_chunk(chunk):
                raw = max(0.0, raw - penalty)
            result.append(chunk.model_copy(update={"rerank_score": raw}))
        return result

    @staticmethod
    def _is_list_caption_chunk(chunk: RetrievedChunk) -> bool:
        """Return True when chunk content is a VLM-generated bullet list (hallucination pattern)."""
        content = chunk.content or ""
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) < 4:
            return False
        bullet_re = __import__("re").compile(r"^\s*[-*•]\s*(\*{0,2})(.+?)(\*{0,2})\s*$")
        short_bullets = sum(
            1 for l in lines
            if (m := bullet_re.match(l)) and len(m.group(2).strip().split()) <= 3
        )
        return (short_bullets / len(lines)) > 0.70

    @staticmethod
    def _rank_key(chunk: RetrievedChunk) -> float:
        rerank = chunk.rerank_score if chunk.rerank_score is not None else 0.0
        # fused_score is RRF-normalized (0–1); blend at 10% weight as tiebreaker only
        return rerank + 0.1 * (chunk.fused_score or 0.0)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-level Jaccard similarity for MMR redundancy measurement."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
