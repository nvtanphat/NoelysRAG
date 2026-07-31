from __future__ import annotations

import json

from evaluation.harness.adaptation.calibration import CalibrationPoint, ThresholdCalibrator
from evaluation.harness.adaptation.dataset_builder import ModelAdaptationDatasetBuilder
from evaluation.harness.adaptation.hard_negative_mining import HardNegativeMiner, TextDocument


def test_hard_negative_miner_prefers_lexically_related_non_positive() -> None:
    miner = HardNegativeMiner()
    negatives = miner.mine(
        query="dropout overfitting validation",
        positive="dropout reduces overfitting by disabling activations",
        candidates=[
            TextDocument(text="batch normalization improves validation stability", metadata={"id": "hard"}),
            TextDocument(text="ancient history chapter", metadata={"id": "easy"}),
            TextDocument(text="dropout reduces overfitting by disabling activations", metadata={"id": "positive"}),
        ],
        limit=1,
    )

    assert negatives[0].metadata["id"] == "hard"


def test_model_adaptation_builder_creates_retrieval_pairs_and_instruction_examples(tmp_path) -> None:
    examples = [
        {
            "id": "ex-1",
            "query": "What is dropout?",
            "expected_answer": "Dropout disables activations.",
            "expected_evidence": [
                {"doc_id": "doc-1", "page": 1, "block_id": "blk-1", "snippet_original": "Dropout disables activations."}
            ],
        },
        {
            "id": "ex-2",
            "query": "What is batch normalization?",
            "expected_answer": "Batch normalization normalizes layer inputs.",
            "expected_evidence": [
                {"doc_id": "doc-2", "page": 2, "block_id": "blk-2", "snippet_original": "Batch normalization normalizes inputs."}
            ],
        },
    ]
    builder = ModelAdaptationDatasetBuilder(hard_negatives_per_query=1)

    pairs = builder.build_retrieval_pairs(examples)
    instructions = builder.build_instruction_examples(examples)
    count = builder.write_jsonl(tmp_path / "pairs.jsonl", pairs)

    assert count == len(pairs)
    assert pairs
    assert pairs[0].query
    assert instructions[0].messages[0]["role"] == "system"
    assert json.loads((tmp_path / "pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])["positive_doc"]


def test_threshold_calibrator_selects_best_f1_threshold() -> None:
    points = [
        CalibrationPoint(score=0.9, is_relevant=True),
        CalibrationPoint(score=0.7, is_relevant=True),
        CalibrationPoint(score=0.4, is_relevant=False),
        CalibrationPoint(score=0.2, is_relevant=False),
    ]

    report = ThresholdCalibrator().calibrate(points, [0.3, 0.5, 0.8])

    assert report.threshold == 0.5
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
