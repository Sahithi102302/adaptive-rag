from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
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




NUM_EXAMPLES = 100
CANDIDATE_K = 20
FINAL_K = 10

OUTPUT_DIR = Path("artifacts/diagnosis")
OUTPUT_PATH = OUTPUT_DIR / "diagnostic_dataset_100.csv"


def get_gold_evidence(
    example,
) -> set[tuple[str, int]]:
    """
    Extract valid HotpotQA gold evidence keys.

    Gold evidence is used only to construct offline oracle labels.
    It must never become an inference-time predictor feature.
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


def main() -> None:

    print("=" * 90)
    print("PHASE 9.2 — DIAGNOSTIC DATASET CONSTRUCTION")
    print("=" * 90)

    # ---------------------------------------------------------
    # Same development slice used in Phase 8
    # ---------------------------------------------------------

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    corpus = build_hotpotqa_corpus(
        dataset["train"],
        max_examples=NUM_EXAMPLES,
    )

    print(
        f"QA examples: {len(corpus.examples):,}"
    )

    print(
        f"Unique documents: {len(corpus.documents):,}"
    )

    # ---------------------------------------------------------
    # Same sentence-aware chunking
    # ---------------------------------------------------------

    print("\nSentence-aware chunking...")

    chunker = SentenceChunker(
        SentenceChunkingConfig(
            sentences_per_chunk=3,
            sentence_overlap=1,
        )
    )

    chunks = chunker.chunk_documents(
        corpus.documents
    )

    print(
        f"Chunks: {len(chunks):,}"
    )

    retrieval_texts = [
        format_chunk_for_retrieval(chunk)
        for chunk in chunks
    ]

    # ---------------------------------------------------------
    # Same dense retrieval configuration
    # ---------------------------------------------------------

    print("\nLoading dense embedder...")

    embedder = DenseEmbedder(
        DenseEmbeddingConfig(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    print(
        f"Embedding {len(retrieval_texts):,} chunks..."
    )

    embeddings = embedder.encode_documents(
        retrieval_texts,
        show_progress_bar=True,
    )

    dense_retriever = DenseRetriever(
        chunks=chunks,
        embeddings=embeddings,
    )

    # ---------------------------------------------------------
    # Same BM25 configuration
    # ---------------------------------------------------------

    print("\nBuilding BM25 index...")

    sparse_retriever = BM25Retriever(
        chunks=chunks,
        retrieval_texts=retrieval_texts,
    )

    # ---------------------------------------------------------
    # Same cross-encoder configuration
    # ---------------------------------------------------------

    print("\nLoading cross-encoder...")

    reranker = CrossEncoderReranker(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config=RerankerConfig(
            model_name=(
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            candidate_k=CANDIDATE_K,
            batch_size=32,
        ),
    )

    # ---------------------------------------------------------
    # Construct rows
    # ---------------------------------------------------------

    rows: list[dict] = []

    print(
        "\nExtracting runtime signals "
        "and oracle labels...\n"
    )

    for index, example in enumerate(
        corpus.examples,
        start=1,
    ):

        gold = get_gold_evidence(example)

        if not gold:
            continue

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        results = reranker.search(
            query=example.question,
            embedder=embedder,
            top_k=FINAL_K,
        )

        # -----------------------------------------------------
        # Runtime features
        #
        # IMPORTANT:
        # This function receives retrieval results only.
        # It never receives gold evidence.
        # -----------------------------------------------------

        runtime_signals = extract_runtime_signals(
            results
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

        lexical_features = asdict(
            lexical_signals
            )

        # -----------------------------------------------------
        # Offline oracle label
        #
        # Gold enters only here.
        # -----------------------------------------------------

        oracle_diagnosis = (
            diagnose_oracle_retrieval(
                gold_evidence=gold,
                results=results,
            )
        )

        is_evidence_sufficient = (
            oracle_diagnosis.failure_type
            == FailureType.SUFFICIENT_EVIDENCE
        )

        row = {
            "question_index": index - 1,
            "question": example.question,

            **runtime_features,
            **lexical_features,
            **semantic_features,

            "oracle_failure_type": (
                oracle_diagnosis.failure_type.value
            ),

            "is_evidence_sufficient": (
                int(is_evidence_sufficient)
            ),
        }

        rows.append(row)

        if (
            index % 10 == 0
            or index == len(corpus.examples)
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(corpus.examples)}"
            )

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------

    diagnostic_df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Leakage guard
    # ---------------------------------------------------------

    forbidden_feature_names = {
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

    leaked_columns = (
        forbidden_feature_names
        & set(diagnostic_df.columns)
    )

    assert set(runtime_features).isdisjoint(
    forbidden_feature_names
)

    assert set(lexical_features).isdisjoint(
    forbidden_feature_names
)
    assert set(semantic_features).isdisjoint(
    forbidden_feature_names
)

    assert not leaked_columns, (
        "Gold-dependent columns leaked into "
        f"runtime dataset: {sorted(leaked_columns)}"
    )

    # ---------------------------------------------------------
    # Consistency checks
    # ---------------------------------------------------------

    assert len(diagnostic_df) == 100

    sufficient_count = int(
        diagnostic_df[
            "is_evidence_sufficient"
        ].sum()
    )

    incomplete_count = (
        len(diagnostic_df)
        - sufficient_count
    )

    assert sufficient_count == 75
    assert incomplete_count == 25

    failure_counts = (
        diagnostic_df[
            "oracle_failure_type"
        ]
        .value_counts()
        .to_dict()
    )

    assert (
        failure_counts.get(
            FailureType.SUFFICIENT_EVIDENCE.value,
            0,
        )
        == 75
    )

    assert (
        failure_counts.get(
            FailureType.PASSAGE_SELECTION_FAILURE.value,
            0,
        )
        == 4
    )

    assert (
        failure_counts.get(
            FailureType.EVIDENCE_COVERAGE_FAILURE.value,
            0,
        )
        == 21
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("DIAGNOSTIC DATASET SUMMARY")
    print("=" * 90)

    print(
        f"\nRows: {len(diagnostic_df):,}"
    )

    print(
        f"Columns: {len(diagnostic_df.columns):,}"
    )

    print(
        f"Sufficient: {sufficient_count}"
    )

    print(
        f"Insufficient: {incomplete_count}"
    )

    print("\nOracle failure distribution:")

    for failure_type, count in (
        diagnostic_df[
            "oracle_failure_type"
        ]
        .value_counts()
        .items()
    ):
        print(
            f"  {failure_type}: {count}"
        )

    print("\nRuntime feature columns:")

    excluded_columns = {
        "question_index",
        "question",
        "oracle_failure_type",
        "is_evidence_sufficient",
    }

    runtime_columns = [
        column
        for column in diagnostic_df.columns
        if column not in excluded_columns
    ]

    for column in runtime_columns:
        print(
            f"  {column}"
        )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print(
        "\nPASS: No gold-dependent feature "
        "columns entered the runtime feature set."
    )

    print(
        "PASS: Reproduced Phase 8.4 "
        "oracle split (75 sufficient / 25 insufficient)."
    )


if __name__ == "__main__":
    main()