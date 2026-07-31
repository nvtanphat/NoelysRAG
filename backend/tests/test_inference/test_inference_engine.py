from __future__ import annotations

import asyncio
import unicodedata

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.inference.inference_engine import InferenceEngine
from src.inference.response_parser import ResponseParser
from src.processing.types import BBox, EvidenceBlock
from src.rag.query_router import PreferredModality
from src.rag.types import RetrievalScope, RetrievedChunk, RetrievedVisualChunk
from src.schemas.query import SentenceCoverageReport, SentenceSupport


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None, preferred_modality: str | None = None, sparse_enabled: bool = True):
        self.calls.append({"query": query, "limit": limit, "preferred_modality": preferred_modality, "sparse_enabled": sparse_enabled})
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                owner_id=scope.owner_id,
                collection_id=scope.collection_id or "c",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                content="Dropout randomly disables activations to reduce co-adaptation.",
                language="en",
                modality="text",
                source_block_ids=["blk-1"],
                source_pages=[4],
                evidence=[
                    EvidenceBlock(
                        owner_id=scope.owner_id,
                        collection_id=scope.collection_id or "c",
                        material_id="65f000000000000000000001",
                        document_name="lecture.pdf",
                        page=4,
                        block_id="blk-1",
                        block_type="paragraph",
                        snippet_original="Dropout randomly disables activations to reduce co-adaptation.",
                        source_language="en",
                        bbox=BBox(x1=1, y1=2, x2=3, y2=4),
                        confidence=0.95,
                    )
                ],
                fused_score=0.8,
            )
        ]


class EmptyRetriever:
    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None, preferred_modality: str | None = None):
        return []


class FakeGraphRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve_paths(self, *, query: str, scope: RetrievalScope, max_hops: int | None = None):
        self.calls.append({"query": query, "max_hops": max_hops})
        return []


class FakeReranker:
    def rerank(self, *, query: str, chunks, limit: int | None = None):
        return [chunk.model_copy(update={"rerank_score": 0.9}) for chunk in chunks][: limit or 5]


class FakeLLM(BaseLLM):
    async def generate(self, *, prompt: str) -> str:
        assert "Dropout randomly disables activations" in prompt
        return "Dropout giảm overfitting bằng cách vô hiệu hóa ngẫu nhiên activation [Nguồn: lecture.pdf, trang 4, block blk-1]."


def test_inference_engine_returns_grounded_answer_with_citation() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=FakeRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Dropout giúp giảm overfitting như thế nào?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert response.answer_language == "vi"
    assert response.citations[0].doc_name == "lecture.pdf"
    assert response.citations[0].block_id == "blk-1"
    assert response.citations[0].bbox.x1 == 1


def test_inference_engine_respects_requested_answer_language() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=FakeRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="How does dropout reduce overfitting?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
            answer_language="vi",
        )
    )

    assert response.answer_language == "vi"


def test_inference_engine_refuses_without_evidence() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=EmptyRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    # Use a knowledge query (domain signal present) so intent classifier routes to RAG,
    # but EmptyRetriever returns nothing → engine should refuse for lack of evidence.
    response = asyncio.run(
        engine.answer(query="Giải thích khái niệm này", scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"))
    )

    assert response.was_refused is True
    assert response.citations == []
    assert "du bang chung" in strip_accents(response.answer.lower())


def test_inference_engine_routes_factual_without_graph_and_scaled_limit() -> None:
    retriever = FakeRetriever()
    graph = FakeGraphRetriever()
    engine = InferenceEngine(
        settings=Settings(testing=True, rerank_input_k=16, final_top_k=5),
        retriever=retriever,
        graph_retriever=graph,
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Dropout là gì?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert retriever.calls[0]["limit"] == 12
    assert graph.calls == []


def test_inference_engine_routes_graph_relation_with_graph() -> None:
    retriever = FakeRetriever()
    graph = FakeGraphRetriever()
    engine = InferenceEngine(
        settings=Settings(testing=True, rerank_input_k=16, final_top_k=5),
        retriever=retriever,
        graph_retriever=graph,
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Quan hệ giữa dropout và overfitting là gì?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert retriever.calls[0]["limit"] == 24
    assert len(graph.calls) == 1


def test_refine_citation_blocks_uses_slec_supporting_block_provenance() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        content="intro block\nexact supporting block",
        language="en",
        modality="text",
        source_block_ids=["blk-intro", "blk-support"],
        source_pages=[1, 2],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=1,
                block_id="blk-intro",
                block_type="paragraph",
                snippet_original="intro block with generic background",
                source_language="en",
                bbox=BBox(x1=1, y1=1, x2=2, y2=2),
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=2,
                block_id="blk-support",
                block_type="paragraph",
                snippet_original="exact supporting block with enough detail for citation",
                source_language="en",
                bbox=BBox(x1=3, y1=4, x2=5, y2=6),
            ),
        ],
        fused_score=0.8,
    )
    citations = ResponseParser().citations_from_chunks(
        [chunk],
        focus_text="intro",
    )
    report = SentenceCoverageReport(
        sentences=[
            SentenceSupport(
                index=0,
                text="Answer sentence [1].",
                status="supported",
                score=0.9,
                supporting_block_ids=["blk-support"],
                citation_refs=[1],
            )
        ]
    )

    refined = InferenceEngine._refine_citation_blocks(citations, [chunk], report)

    assert refined[0].block_id == "blk-support"
    assert refined[0].page == 2
    assert refined[0].bbox.x1 == 3
    assert refined[0].snippet_original.startswith("exact supporting block")


def test_prune_to_cited_rewrites_sentence_coverage_text_markers() -> None:
    report = SentenceCoverageReport(
        sentences=[
            SentenceSupport(
                index=0,
                text="First sentence [2].",
                status="supported",
                score=0.9,
                citation_refs=[2],
            ),
            SentenceSupport(
                index=1,
                text="Second sentence [4].",
                status="supported",
                score=0.9,
                citation_refs=[4],
            ),
        ]
    )
    citations = [object(), object(), object(), object()]

    answer, pruned, updated_report, _ = InferenceEngine._prune_to_cited(
        "First sentence [2]. Second sentence [4].",
        citations,
        report,
    )

    assert answer == "First sentence [1]. Second sentence [2]."
    assert len(pruned) == 2
    assert updated_report.sentences[0].text == "First sentence [1]."
    assert updated_report.sentences[0].citation_refs == [1]
    assert updated_report.sentences[1].text == "Second sentence [2]."
    assert updated_report.sentences[1].citation_refs == [2]


def test_visual_verifier_refusal_requires_image_and_high_confidence() -> None:
    class Verdict:
        supported = False
        confidence = 0.4

    assert InferenceEngine._visual_verifier_should_refuse(
        visual_verdict=Verdict(),
        image_paths=["figure.png"],
        threshold=0.75,
    ) is False
    assert InferenceEngine._visual_verifier_should_refuse(
        visual_verdict=Verdict(),
        image_paths=[],
        threshold=0.75,
    ) is False

    Verdict.confidence = 0.9
    assert InferenceEngine._visual_verifier_should_refuse(
        visual_verdict=Verdict(),
        image_paths=["figure.png"],
        threshold=0.75,
    ) is True


def test_inline_visual_alt_text_is_short_label_and_filter_respects_figure_number() -> None:
    hit1 = RetrievedVisualChunk(
        point_id="v1",
        owner_id="u",
        collection_id="c",
        material_id="m",
        document_name="paper.pdf",
        page=3,
        block_id="fig-1",
        block_type="figure",
        caption="Figure 1: " + "very long caption " * 40,
        source_language="en",
        score=0.9,
    )
    hit2 = hit1.model_copy(update={"point_id": "v2", "block_id": "fig-2", "page": 4, "caption": "Figure 2: other"})

    filtered = InferenceEngine._filter_visual_hits_for_query("Describe Figure 1", [hit2, hit1])
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=FakeRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )
    answer = engine._inject_inline_images(answer="Answer [1].", visual_hits=filtered, owner_id="u")
    citation = engine._visual_hit_to_citation(hit1)

    assert filtered == [hit1]
    assert "![Figure 1, trang 3]" in answer
    assert "very long caption" not in answer
    assert citation.role == "visual_match"
    assert citation.evidence_id == "v1"
    assert citation.kind == "visual"
    assert citation.evidence_blocks


def test_non_figure_answers_strip_inline_image_markdown() -> None:
    answer = "Cau tra loi text [1].\n\n![Hinh lac de](/api/v1/materials/m/raw?owner_id=u)\n\nCau tiep [1]."

    cleaned = InferenceEngine._strip_inline_image_markdown(answer)

    assert "![" not in cleaned
    assert "Cau tra loi text [1]." in cleaned
    assert "Cau tiep [1]." in cleaned
    assert InferenceEngine._allows_visual_answer_content(PreferredModality.NONE) is False
    assert InferenceEngine._allows_visual_answer_content(PreferredModality.FIGURE) is True


def test_citation_primary_can_be_short_matching_label() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-short-label",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        content="THANH TOAN\nTRI TUE NHAN TAO DANH CHO MOI NGUOI",
        language="vi",
        modality="text",
        source_block_ids=["blk-label", "blk-title"],
        source_pages=[87, 91],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=87,
                block_id="blk-label",
                block_type="paragraph",
                snippet_original="THANH TOAN",
                source_language="vi",
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=91,
                block_id="blk-title",
                block_type="paragraph",
                snippet_original="TRI TUE NHAN TAO DANH CHO MOI NGUOI ThS NGUYEN NGOC TU",
                source_language="vi",
            ),
        ],
        fused_score=0.8,
    )

    citations = ResponseParser().citations_from_chunks([chunk], focus_text="nhan THANH TOAN")

    assert citations[0].block_id == "blk-label"
    assert citations[0].page == 87
    assert citations[0].snippet_original == "THANH TOAN"


def test_standalone_label_query_can_be_answered_deterministically() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-label",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        content="THANH TOAN\nTRI TUE NHAN TAO DANH CHO MOI NGUOI",
        language="vi",
        modality="text",
        source_block_ids=["blk-label", "blk-title"],
        source_pages=[90, 91],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=90,
                block_id="blk-label",
                block_type="paragraph",
                snippet_original="THANH TOAN",
                source_language="vi",
            ),
        ],
        fused_score=0.8,
    )

    answer = InferenceEngine._maybe_answer_standalone_label_query(
        query='Trong đoạn trích, chức năng/nhãn nào xuất hiện cùng với "THANH TOAN"?',
        chunks=[chunk],
        answer_language="vi",
    )

    assert answer == 'Chức năng/nhãn được nêu là “THANH TOAN” [1].'


def test_slide_citation_snippet_expands_adjacent_short_blocks() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-ai-tools",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        content="AI CO NHIEU CONG CU\nHoc may\nHoc sau",
        language="vi",
        modality="heading",
        source_block_ids=["blk-title", "blk-ml", "blk-dl"],
        source_pages=[53],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=53,
                block_id="blk-title",
                block_type="heading",
                snippet_original="AI CO NHIEU CONG CU",
                source_language="vi",
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=53,
                block_id="blk-ml",
                block_type="paragraph",
                snippet_original="Hoc may va Khoa hoc du lieu",
                source_language="vi",
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=53,
                block_id="blk-dl",
                block_type="paragraph",
                snippet_original="Hoc sau va Mang no-ron",
                source_language="vi",
            ),
        ],
        fused_score=0.8,
    )

    citations = ResponseParser().citations_from_chunks([chunk], focus_text="AI co cong cu nao")

    assert citations[0].block_id == "blk-title"
    assert "Hoc may va Khoa hoc du lieu" in citations[0].snippet_original
    assert "Hoc sau va Mang no-ron" in citations[0].snippet_original


def test_primary_citation_prefers_text_over_picture_for_factual_query() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-marker",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="paper.pdf",
        content="Marker extracts PDF content and improves speed and accuracy.",
        language="en",
        modality="mixed",
        source_block_ids=["pic-1", "txt-1"],
        source_pages=[4],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=4,
                block_id="pic-1",
                block_type="picture",
                snippet_original=(
                    "Marker PDF speed accuracy " + "background preprocessing text " * 40
                ),
                source_language="en",
                confidence=0.9,
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=4,
                block_id="txt-1",
                block_type="paragraph",
                snippet_original="Marker extracts PDF content and improves speed and accuracy.",
                source_language="en",
                confidence=0.9,
            ),
        ],
        fused_score=0.8,
    )

    citations = ResponseParser().citations_from_chunks(
        [chunk],
        focus_text="Marker purpose PDF improve speed accuracy",
    )

    assert citations[0].block_id == "txt-1"
    assert citations[0].block_type == "paragraph"


def test_long_citation_snippet_is_focused_around_query_terms() -> None:
    noisy_prefix = "Header footer preprocessing noise. " * 80
    marker_text = (
        "Marker is designed for identifying and extracting content from PDF "
        "documents and utilizes models to enhance speed and accuracy."
    )
    chunk = RetrievedChunk(
        chunk_id="chunk-long-ocr",
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="paper.pdf",
        content=noisy_prefix + marker_text,
        language="en",
        modality="figure",
        source_block_ids=["pic-1"],
        source_pages=[4],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=4,
                block_id="pic-1",
                block_type="figure",
                snippet_original=noisy_prefix + marker_text,
                source_language="en",
            )
        ],
        fused_score=0.8,
    )

    citations = ResponseParser().citations_from_chunks(
        [chunk],
        focus_text="Marker purpose PDF improve speed accuracy",
    )

    assert "Marker is designed" in citations[0].snippet_original
    assert "speed and accuracy" in citations[0].snippet_original
    assert len(citations[0].snippet_original) <= 1010
