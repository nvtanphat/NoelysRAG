from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluation.harness.adaptation.hard_negative_mining import HardNegativeMiner, TextDocument


@dataclass(frozen=True)
class RetrievalPair:
    query: str
    positive_doc: str
    negative_doc: str
    positive_evidence: dict
    negative_evidence: dict


@dataclass(frozen=True)
class InstructionExample:
    messages: list[dict[str, str]]
    metadata: dict


class ModelAdaptationDatasetBuilder:
    def __init__(self, *, hard_negatives_per_query: int = 2) -> None:
        self.hard_negatives_per_query = hard_negatives_per_query
        self.miner = HardNegativeMiner()

    def build_retrieval_pairs(self, examples: list[dict]) -> list[RetrievalPair]:
        documents = self._documents_from_examples(examples)
        pairs: list[RetrievalPair] = []
        for example in examples:
            query = self._query_text(example)
            positives = example.get("expected_evidence", [])
            if not query or not positives:
                continue
            positive = positives[0]
            positive_doc = self._evidence_text(positive, fallback=example.get("expected_answer", ""))
            negatives = self.miner.mine(query=query, positive=positive_doc, candidates=documents, limit=self.hard_negatives_per_query)
            for negative in negatives:
                pairs.append(
                    RetrievalPair(
                        query=query,
                        positive_doc=positive_doc,
                        negative_doc=negative.text,
                        positive_evidence=positive,
                        negative_evidence=negative.metadata,
                    )
                )
        return pairs

    def build_instruction_examples(self, examples: list[dict]) -> list[InstructionExample]:
        instructions: list[InstructionExample] = []
        for example in examples:
            query = self._query_text(example)
            expected_answer = example.get("expected_answer") or example.get("expected_answer_vi")
            evidence = example.get("expected_evidence", [])
            if not query or not expected_answer or not evidence:
                continue
            context = "\n".join(self._evidence_text(item, fallback="") for item in evidence)
            instructions.append(
                InstructionExample(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are AgentBook. Answer only from Context and cite document/page/block.",
                        },
                        {"role": "user", "content": f"Question: {query}\nContext: {context}"},
                        {"role": "assistant", "content": expected_answer},
                    ],
                    metadata={"example_id": example.get("id"), "evidence": evidence},
                )
            )
        return instructions

    @staticmethod
    def load_examples(paths: Iterable[Path]) -> list[dict]:
        examples: list[dict] = []
        for path in paths:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                examples.extend(data)
        return examples

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[object]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if hasattr(row, "__dataclass_fields__"):
                    payload = row.__dict__
                else:
                    payload = row
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def _query_text(example: dict) -> str:
        return str(example.get("query") or example.get("question_vi") or example.get("claim") or "").strip()

    @staticmethod
    def _evidence_text(evidence: dict, *, fallback: str) -> str:
        for key in ("snippet_original", "text", "content", "expected_answer"):
            value = evidence.get(key)
            if value:
                return str(value)
        return fallback or f"{evidence.get('document_name', evidence.get('doc_id', 'document'))} page {evidence.get('page')} block {evidence.get('block_id')}"

    def _documents_from_examples(self, examples: list[dict]) -> list[TextDocument]:
        documents: list[TextDocument] = []
        for example in examples:
            for evidence in example.get("expected_evidence", []):
                documents.append(
                    TextDocument(
                        text=self._evidence_text(evidence, fallback=example.get("expected_answer", "")),
                        metadata=evidence,
                    )
                )
            expected_answer = example.get("expected_answer") or example.get("expected_answer_vi")
            if expected_answer:
                documents.append(TextDocument(text=str(expected_answer), metadata={"example_id": example.get("id"), "kind": "answer"}))
        return documents
