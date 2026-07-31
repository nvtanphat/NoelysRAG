from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.agentic.planner import AgenticPlanner
from src.agentic.service import AgenticRagService
from src.guardrails.claim_verifier import ClaimVerificationResult, ClaimVerdict
from src.inference.intent_classifier import QueryIntent
from src.inference.quality_finalizer import QualityFinalizeResult
from src.processing.types import EvidenceBlock
from src.rag.query_router import QueryRouter, RouteType
from src.rag.types import RetrievalScope, RetrievedChunk


def _chunk(material_id: str, content: str = "Dropout reduces overfitting.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{material_id[-1]}",
        owner_id="user_demo",
        collection_id="65f000000000000000000010",
        material_id=material_id,
        document_name=f"{material_id[-1]}.pdf",
        content=content,
        language="en",
        modality="text",
        source_block_ids=[f"blk-{material_id[-1]}"],
        source_pages=[1],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000010",
                material_id=material_id,
                document_name=f"{material_id[-1]}.pdf",
                page=1,
                block_id=f"blk-{material_id[-1]}",
                block_type="paragraph",
                snippet_original=content,
                source_language="en",
                confidence=0.95,
            )
        ],
        fused_score=0.8,
    )


class FakeIntentClassifier:
    async def classify(self, query: str) -> QueryIntent:
        return QueryIntent.KNOWLEDGE


class FakeQueryProcessor:
    async def process_async(self, query: str, *, answer_language: str | None = None, rewriter=None):
        return SimpleNamespace(
            retrieval_queries=[query, f"{query} evidence"],
            answer_language=answer_language or "vi",
            query_language="vi",
            translated_query=None,
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None, preferred_modality: str | None = None):
        self.calls.append((query, list(scope.material_ids)))
        material_ids = scope.material_ids or ["65f000000000000000000001", "65f000000000000000000002"]
        return [_chunk(material_id) for material_id in material_ids[: max(1, min(len(material_ids), limit or 2))]]


class FakeGraphRetriever:
    async def retrieve_paths(self, *, query: str, scope: RetrievalScope, max_hops: int | None = None):
        return []


class FakeReranker:
    def rerank_multilingual(self, *, queries, chunks, limit: int, use_mmr: bool):
        return [chunk.model_copy(update={"rerank_score": 0.9}) for chunk in chunks][:limit]


class FakeConfidenceScorer:
    def score(self, chunks) -> float:
        return 0.9 if chunks else 0.0

    def should_refuse(self, *, chunks, confidence: float):
        return (not chunks, "no evidence" if not chunks else None)


class FakeResponseParser:
    def format_evidence_for_prompt(self, chunks):
        return "\n\n".join(chunk.content for chunk in chunks)

    def citations_from_chunks(self, chunks, focus_text: str | None = None):
        return []

    def inject_citations(self, answer: str, chunks) -> str:
        return answer


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, prompt: str) -> str:
        self.calls += 1
        return "Dropout giam overfitting dua tren bang chung."


class FakeClaimVerifier:
    def verify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            verdict=ClaimVerdict.SUPPORTED,
            citations=evidence,
            confidence=0.75,
            was_refused=False,
        )


class FakeCRAGEvaluator:
    def evaluate(self, *, chunks):
        return chunks


class RepairingClaimVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        self.calls += 1
        verdict = ClaimVerdict.NOT_ENOUGH_EVIDENCE if self.calls == 1 else ClaimVerdict.SUPPORTED
        return ClaimVerificationResult(
            verdict=verdict,
            citations=evidence,
            confidence=0.65,
            was_refused=False,
        )


class FakeEngine:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            rerank_input_k=4,
            final_top_k=4,
            crag_correct_threshold=0.55,
            crag_incorrect_threshold=0.25,
            agentic_max_retrieval_iterations=2,
            agentic_critic_activation_confidence=0.65,
            agentic_anaphora_resolution_enabled=False,
            agentic_planner_llm_enabled=False,
            multi_query_enabled=False,
            visual_embedding_enabled=False,
        )
        self.intent_classifier = FakeIntentClassifier()
        self.query_router = QueryRouter()
        self.query_rewriter = object()
        self.query_processor = FakeQueryProcessor()
        self.retriever = FakeRetriever()
        self.graph_retriever = FakeGraphRetriever()
        self.reranker = FakeReranker()
        self.confidence_scorer = FakeConfidenceScorer()
        self.response_parser = FakeResponseParser()
        self.llm = FakeLLM()
        self.claim_verifier = FakeClaimVerifier()
        self.crag_evaluator = FakeCRAGEvaluator()
        self.visual_provider = None

    async def _answer_chitchat(self, query: str):
        raise AssertionError("not used")

    def _refuse_off_topic(self):
        raise AssertionError("not used")

    def _scaled_limit(self, base: int, decision) -> int:
        return max(1, int(base * decision.top_k_multiplier))

    def _chunks_from_graph_paths(self, graph_paths, *, scope: RetrievalScope, priority: bool = False):
        return []

    def _pack_context_chunks(self, chunks):
        return chunks

    async def _graph_probe(self, *, query: str, chunks, scope: RetrievalScope, route_decision=None, enabled: bool = True):
        return chunks

    async def _finalize_quality(
        self, *, answer: str, citations: list, confidence: float, evidence_bundle, context_chunks: list,
        route_decision=None, trace=None, run_slec: bool = True, multimodal: bool = False,
    ) -> QualityFinalizeResult:
        return QualityFinalizeResult(answer=answer, citations=citations)

    def _build_prompt(self, *, query: str, chunks, answer_language: str, memory_context: str = "", route_type=RouteType.GENERAL, plan_type: str | None = None):
        return f"{query}\n{len(chunks)} chunks\n{memory_context}"


def test_agentic_planner_multi_source_requires_coverage_repair() -> None:
    plan = AgenticPlanner().build(query="So sanh dropout va weight decay", route=QueryRouter().route("So sanh dropout va weight decay"), material_count=2)

    assert plan.use_multi_query is True
    assert plan.use_per_source is True
    assert plan.requires_coverage is True
    assert "repair_retrieval" in plan.steps
    assert "retrieve_sub_questions" in plan.steps
    assert len(plan.sub_questions) >= 3


def test_agentic_planner_creates_route_specific_sub_questions() -> None:
    router = QueryRouter()

    comparison = AgenticPlanner().build(query="Compare dropout and weight decay", route=router.route("Compare dropout and weight decay"), material_count=2)
    claim_check = AgenticPlanner().build(query="Is dropout always better, is it true?", route=router.route("Is dropout always better, is it true?"), material_count=1)
    relation = AgenticPlanner().build(query="How does dropout affect overfitting?", route=router.route("How does dropout affect overfitting?"), material_count=1)

    assert comparison.plan_type == "comparison"
    assert any(item.tool == "retrieve_per_source" for item in comparison.sub_questions)
    assert claim_check.plan_type == "claim_check"
    assert any("contradicts" in item.text for item in claim_check.sub_questions)
    assert relation.plan_type == "relation_trace"
    assert relation.use_graph is True
    assert "trace_graph" in relation.steps
    assert all(item.tool != "trace_graph" for item in relation.sub_questions)
    assert relation.use_per_source is False


def test_agentic_service_returns_trace_coverage_and_emits_steps() -> None:
    engine = FakeEngine()
    service = AgenticRagService(engine=engine)  # type: ignore[arg-type]
    emitted: list[str] = []

    async def on_step(step):
        emitted.append(step.name)

    response = asyncio.run(
        service.answer(
            query="So sanh dropout va weight decay",
            scope=RetrievalScope(
                owner_id="user_demo",
                collection_id="65f000000000000000000010",
                material_ids=["65f000000000000000000001", "65f000000000000000000002"],
            ),
            on_step=on_step,
        )
    )

    assert response.was_refused is False
    assert response.agent_trace is not None
    assert response.agent_trace.plan_type == "comparison"
    assert response.coverage is not None
    assert response.coverage.covered_count == 2
    assert "plan_query" in emitted
    # New coordinating engine merges per-source retrieval into the
    # RetrieverDirector — the explicit step is "retrieve_evidence".
    assert "retrieve_evidence" in emitted
    assert "crag_triage" in emitted
    assert "rerank_evidence" in emitted
    assert "verify_claims" in emitted


def test_agentic_service_repairs_weak_answer_once() -> None:
    engine = FakeEngine()
    engine.claim_verifier = RepairingClaimVerifier()
    service = AgenticRagService(engine=engine)  # type: ignore[arg-type]

    response = asyncio.run(
        service.answer(
            query="Dropout co dung la giam overfitting khong?",
            scope=RetrievalScope(
                owner_id="user_demo",
                collection_id="65f000000000000000000010",
                material_ids=["65f000000000000000000001"],
            ),
        )
    )

    assert response.was_refused is False
    assert response.agent_trace is not None
    assert any(step.name == "repair_answer" for step in response.agent_trace.steps)
    assert response.agent_trace.verification is not None
    assert response.agent_trace.verification.repair_attempted is True
    assert engine.llm.calls == 2
