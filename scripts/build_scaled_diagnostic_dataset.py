from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.chunking.chunk import Chunk
from src.diagnosis.evidence_signals import (
    extract_lexical_evidence_signals,
    extract_semantic_evidence_signals,
)
from src.diagnosis.oracle import diagnose_oracle_retrieval
from src.diagnosis.signals import extract_runtime_signals
from src.diagnosis.states import FailureType
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.ingestion.hotpotqa import build_hotpotqa_corpus
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)
from src.retrieval.sparse_retriever import BM25Retriever


# =========================================================
# Configuration
# =========================================================

ARTIFACT_DIR = Path("artifacts/diagnosis/scaled")

CHUNKS_PATH = ARTIFACT_DIR / "pooled_chunks.jsonl"
EMBEDDINGS_PATH = ARTIFACT_DIR / "pooled_embeddings.npy"

CANDIDATE_K = 20
FINAL_K = 10

EXPECTED_CHUNKS = 32368
EXPECTED_DIMENSION = 384

DENSE_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

CROSS_ENCODER_MODEL_NAME = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)


# =========================================================
# Frozen query splits
# =========================================================

SPLIT_CONFIGS = {
    "train": {
        "dataset_split": "train",
        "start": 100,
        "end": 1100,
        "expected_count": 1000,
    },
    "validation": {
        "dataset_split": "train",
        "start": 1100,
        "end": 1300,
        "expected_count": 200,
    },
    "test": {
        "dataset_split": "validation",
        "start": 0,
        "end": 500,
        "expected_count": 500,
    },
}


# =========================================================
# Leakage protection
# =========================================================

FORBIDDEN_FEATURE_NAMES = {
    "gold_fact_count",
    "retrieved_gold_fact_count",
    "gold_document_count",
    "retrieved_gold_document_count",
    "has_any_gold_evidence",
    "has_partial_gold_evidence",
    "has_complete_gold_evidence",
    "has_missing_gold_document",
    "has_gold_document_wrong_passage",
    "missing_gold_fact_count",
    "missing_gold_document_count",
}


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_CONFIGS),
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Example: --limit 10"
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete the selected split's existing "
            "checkpoint before starting."
        ),
    )

    return parser.parse_args()


# =========================================================
# Frozen corpus loading
# =========================================================

def load_frozen_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            record = json.loads(line)

            chunks.append(
                Chunk(
                    id=record["id"],
                    document_id=record["document_id"],
                    chunk_index=record["chunk_index"],
                    text=record["text"],
                    start_char=record["start_char"],
                    end_char=record["end_char"],
                    metadata=record["metadata"],
                )
            )

    return chunks


# =========================================================
# Gold evidence
# =========================================================

def get_gold_evidence(
    example,
) -> set[tuple[str, int]]:
    """
    Extract valid HotpotQA gold evidence.

    IMPORTANT:
    Gold is used only for the offline oracle target.
    It must never become an inference-time feature.
    """

    return {
        (
            fact.document_id,
            fact.sentence_id,
        )
        for fact in example.supporting_facts
        if (
            fact.document_id is not None
            and fact.sentence_id >= 0
        )
    }


# =========================================================
# Query loading
# =========================================================

def load_query_examples(
    split_name: str,
):
    config = SPLIT_CONFIGS[split_name]

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    raw_split = dataset[
        config["dataset_split"]
    ]

    selected_raw = raw_split.select(
        range(
            config["start"],
            config["end"],
        )
    )

    corpus = build_hotpotqa_corpus(
        selected_raw
    )

    assert (
        len(corpus.examples)
        == config["expected_count"]
    ), (
        "Unexpected query count for "
        f"{split_name}: "
        f"{len(corpus.examples)}"
    )

    return corpus.examples


# =========================================================
# Artifact paths
# =========================================================

def checkpoint_path(
    split_name: str,
) -> Path:
    return (
        ARTIFACT_DIR
        / f"diagnostic_{split_name}_checkpoint.csv"
    )


def final_path(
    split_name: str,
) -> Path:
    return (
        ARTIFACT_DIR
        / f"diagnostic_{split_name}.csv"
    )


# =========================================================
# Checkpoint persistence
# =========================================================

def save_checkpoint(
    rows: list[dict],
    path: Path,
) -> None:
    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        path,
        index=False,
    )


# =========================================================
# Leakage validation
# =========================================================

def validate_feature_names(
    runtime_features: dict,
    lexical_features: dict,
    semantic_features: dict,
) -> None:
    all_feature_names = (
        set(runtime_features)
        | set(lexical_features)
        | set(semantic_features)
    )

    leaked = (
        all_feature_names
        & FORBIDDEN_FEATURE_NAMES
    )

    assert not leaked, (
        "Gold-dependent feature names leaked "
        f"into runtime features: {sorted(leaked)}"
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()

    split_name = args.split

    split_config = SPLIT_CONFIGS[
        split_name
    ]

    if args.checkpoint_every <= 0:
        raise ValueError(
            "--checkpoint-every must be positive."
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = checkpoint_path(
        split_name
    )

    output = final_path(
        split_name
    )

    print("=" * 90)
    print(
        "PHASE 9.5E — SCALED "
        "DIAGNOSTIC DATASET"
    )
    print("=" * 90)

    print(
        f"\nSplit: {split_name}"
    )

    print(
        "Source: "
        f"HotpotQA "
        f"{split_config['dataset_split']}"
        f"[{split_config['start']}:"
        f"{split_config['end']}]"
    )

    # ---------------------------------------------------------
    # Optional fresh start
    # ---------------------------------------------------------

    if args.fresh and checkpoint.exists():
        print(
            f"\nRemoving checkpoint: "
            f"{checkpoint}"
        )

        checkpoint.unlink()

    # ---------------------------------------------------------
    # Load exact query split
    # ---------------------------------------------------------

    print(
        "\nLoading query examples..."
    )

    examples = load_query_examples(
        split_name
    )

    full_count = len(examples)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        examples = examples[
            : min(
                args.limit,
                len(examples),
            )
        ]

    target_count = len(examples)

    print(
        f"Full split questions: "
        f"{full_count:,}"
    )

    print(
        f"Questions this run: "
        f"{target_count:,}"
    )

    # ---------------------------------------------------------
    # Load frozen retrieval corpus
    # ---------------------------------------------------------

    print(
        "\nLoading frozen chunks..."
    )

    chunks = load_frozen_chunks()

    assert (
        len(chunks)
        == EXPECTED_CHUNKS
    ), (
        "Frozen chunk count changed. "
        f"Expected {EXPECTED_CHUNKS:,}, "
        f"found {len(chunks):,}."
    )

    print(
        f"Frozen chunks: "
        f"{len(chunks):,}"
    )

    print(
        "Loading persisted embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    assert embeddings.shape == (
        EXPECTED_CHUNKS,
        EXPECTED_DIMENSION,
    ), (
        "Unexpected embedding shape: "
        f"{embeddings.shape}"
    )

    assert (
        embeddings.dtype
        == np.float32
    ), (
        "Unexpected embedding dtype: "
        f"{embeddings.dtype}"
    )

    # ---------------------------------------------------------
    # Shared retrieval representation for BM25
    # ---------------------------------------------------------

    print(
        "\nPreparing BM25 retrieval texts..."
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    # ---------------------------------------------------------
    # Dense retriever
    # ---------------------------------------------------------

    print(
        "Constructing DenseRetriever..."
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print(
        "Building BM25 index..."
    )

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    # ---------------------------------------------------------
    # MiniLM query / semantic encoder
    # ---------------------------------------------------------

    print(
        "Loading dense embedder..."
    )

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=DENSE_MODEL_NAME,
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    assert (
        embedder.embedding_dimension
        == EXPECTED_DIMENSION
    ), (
        "Unexpected embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    # ---------------------------------------------------------
    # Cross-encoder
    # ---------------------------------------------------------

    print(
        "Loading cross-encoder..."
    )

    reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            model_name=(
                CROSS_ENCODER_MODEL_NAME
            ),
            candidate_k=CANDIDATE_K,
            batch_size=32,
        ),
    )

    # ---------------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------------

    rows: list[dict] = []

    if checkpoint.exists():
        existing_df = pd.read_csv(
            checkpoint
        )

        rows = existing_df.to_dict(
            orient="records"
        )

        if len(rows) > target_count:
            raise RuntimeError(
                "Checkpoint contains more rows "
                "than requested by this run. "
                "Use --fresh for a new smoke run."
            )

        print(
            f"\nResuming from checkpoint: "
            f"{len(rows):,} rows"
        )

    start_offset = len(rows)

    # ---------------------------------------------------------
    # Resume-integrity validation
    # ---------------------------------------------------------

    if rows:
        expected_indices = [
            split_config["start"] + offset
            for offset in range(start_offset)
        ]

        actual_indices = [
            int(row["question_index"])
            for row in rows
        ]

        assert (
            actual_indices
            == expected_indices
        ), (
            "Checkpoint is not an ordered prefix "
            f"of the {split_name} split.\n"
            f"Expected indices: "
            f"{expected_indices[:5]}..."
            f"{expected_indices[-5:]}\n"
            f"Actual indices: "
            f"{actual_indices[:5]}..."
            f"{actual_indices[-5:]}"
        )

        actual_splits = {
            str(row["split"])
            for row in rows
        }

        assert (
            actual_splits
            == {split_name}
        ), (
            "Checkpoint split mismatch: "
            f"{actual_splits}"
        )

        print(
            "Checkpoint integrity: PASS "
            f"(ordered prefix of "
            f"{start_offset:,} rows)"
        )

    # ---------------------------------------------------------
    # Construct rows
    # ---------------------------------------------------------

    print(
        "\nExtracting retrieval signals, "
        "evidence signals, and oracle labels..."
    )

    for local_index in range(
        start_offset,
        target_count,
    ):
        example = examples[
            local_index
        ]

        absolute_question_index = (
            split_config["start"]
            + local_index
        )

        gold = get_gold_evidence(
            example
        )

        if not gold:
            raise RuntimeError(
                "Encountered question with "
                "zero valid gold evidence at "
                f"local index {local_index}."
            )

        # -----------------------------------------------------
        # Retrieval + cross-encoder reranking
        # -----------------------------------------------------

        results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=FINAL_K,
        )

        assert (
            len(results)
            == FINAL_K
        ), (
            "Unexpected number of reranked "
            f"results at question "
            f"{absolute_question_index}: "
            f"{len(results)}"
        )

        # -----------------------------------------------------
        # Gold-free runtime features
        # -----------------------------------------------------

        runtime_signals = (
            extract_runtime_signals(
                results
            )
        )

        runtime_features = asdict(
            runtime_signals
        )

        lexical_signals = (
            extract_lexical_evidence_signals(
                query=example.question,
                results=results,
            )
        )

        lexical_features = asdict(
            lexical_signals
        )

        semantic_signals = (
            extract_semantic_evidence_signals(
                query=example.question,
                results=results,
                embedder=embedder,
            )
        )

        semantic_features = asdict(
            semantic_signals
        )

        # -----------------------------------------------------
        # Explicit feature-level leakage guard
        # -----------------------------------------------------

        validate_feature_names(
            runtime_features,
            lexical_features,
            semantic_features,
        )

        # -----------------------------------------------------
        # Offline oracle target
        #
        # IMPORTANT:
        # Gold enters only here.
        # -----------------------------------------------------

        oracle_diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=results,
            )
        )

        is_evidence_sufficient = int(
            oracle_diagnosis.failure_type
            == FailureType.SUFFICIENT_EVIDENCE
        )

        row = {
            "split": split_name,
            "question_index": (
                absolute_question_index
            ),
            "question": (
                example.question
            ),

            **runtime_features,
            **lexical_features,
            **semantic_features,

            "oracle_failure_type": (
                oracle_diagnosis
                .failure_type
                .value
            ),

            "is_evidence_sufficient": (
                is_evidence_sufficient
            ),
        }

        rows.append(row)

        processed = local_index + 1

        if (
            processed
            % args.checkpoint_every
            == 0
            or processed
            == target_count
        ):
            save_checkpoint(
                rows,
                checkpoint,
            )

            print(
                f"Checkpoint: "
                f"{processed:,}/"
                f"{target_count:,}"
            )

    # ---------------------------------------------------------
    # Final dataframe validation
    # ---------------------------------------------------------

    diagnostic_df = pd.DataFrame(
        rows
    )

    assert (
        len(diagnostic_df)
        == target_count
    ), (
        "Final row count mismatch: "
        f"{len(diagnostic_df)} "
        f"!= {target_count}"
    )

    leaked_columns = (
        FORBIDDEN_FEATURE_NAMES
        & set(diagnostic_df.columns)
    )

    assert not leaked_columns, (
        "Gold-dependent columns leaked "
        "into saved diagnostic dataset: "
        f"{sorted(leaked_columns)}"
    )

    assert (
        diagnostic_df[
            "question_index"
        ].is_unique
    ), (
        "Duplicate question_index values "
        "detected."
    )

    expected_final_indices = list(
        range(
            split_config["start"],
            split_config["start"]
            + target_count,
        )
    )

    actual_final_indices = (
        diagnostic_df[
            "question_index"
        ]
        .astype(int)
        .tolist()
    )

    assert (
        actual_final_indices
        == expected_final_indices
    ), (
        "Final dataset indices are not "
        "the expected ordered prefix."
    )

    assert set(
        diagnostic_df["split"]
        .astype(str)
        .unique()
    ) == {
        split_name
    }, (
        "Final dataset contains an "
        "unexpected split value."
    )

    # ---------------------------------------------------------
    # Save behavior
    #
    # Limited runs intentionally retain the checkpoint.
    # Full runs additionally create the final split file.
    # ---------------------------------------------------------

    if args.limit is None:
        diagnostic_df.to_csv(
            output,
            index=False,
        )

        if checkpoint.exists():
            checkpoint.unlink()

        saved_path = output
        run_type = "FULL SPLIT"

    else:
        saved_path = checkpoint
        run_type = "SMOKE / LIMITED"

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    sufficient_count = int(
        diagnostic_df[
            "is_evidence_sufficient"
        ].sum()
    )

    insufficient_count = (
        len(diagnostic_df)
        - sufficient_count
    )

    print("\n" + "=" * 90)
    print(
        "SCALED DIAGNOSTIC DATASET SUMMARY"
    )
    print("=" * 90)

    print(
        f"\nRun type: {run_type}"
    )

    print(
        f"Split: {split_name}"
    )

    print(
        f"Rows: "
        f"{len(diagnostic_df):,}"
    )

    print(
        f"Columns: "
        f"{len(diagnostic_df.columns):,}"
    )

    print(
        f"Sufficient: "
        f"{sufficient_count:,}"
    )

    print(
        f"Insufficient: "
        f"{insufficient_count:,}"
    )

    print(
        "\nOracle failure distribution:"
    )

    for failure_type, count in (
        diagnostic_df[
            "oracle_failure_type"
        ]
        .value_counts()
        .items()
    ):
        print(
            f"  {failure_type}: "
            f"{count}"
        )

    print(
        f"\nSaved to: "
        f"{saved_path}"
    )

    print(
        "\nPASS: No gold-dependent "
        "feature columns entered the "
        "runtime feature set."
    )

    print(
        "PASS: Question indices are "
        "unique and ordered."
    )

    if args.limit is not None:
        print(
            "NOTE: This was a smoke run. "
            "Do not interpret its label "
            "distribution as model performance."
        )


if __name__ == "__main__":
    main()