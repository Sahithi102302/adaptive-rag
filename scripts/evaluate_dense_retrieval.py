from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from datasets import load_dataset

from src.chunking.sentence_chunker import (
    SentenceChunker,
    SentenceChunkingConfig,
)
from src.embeddings.dense_embedder import (
    DenseEmbedder,
    DenseEmbeddingConfig,
)
from src.ingestion.hotpotqa import (
    build_hotpotqa_corpus,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    format_chunk_for_retrieval,
)


# ============================================================
# Configuration
# ============================================================

NUM_EXAMPLES = 100

TOP_K_VALUES = (
    1,
    5,
    10,
)

EvidenceKey = Tuple[str, int]


# ============================================================
# Evaluation accumulator
# ============================================================

@dataclass
class EvaluationTotals:
    questions: int = 0
    valid_gold_facts: int = 0

    reciprocal_rank_sum: float = 0.0

    evidence_found: Dict[int, int] = None
    question_hits: Dict[int, int] = None
    complete_questions: Dict[int, int] = None

    def __post_init__(self) -> None:
        self.evidence_found = {
            k: 0
            for k in TOP_K_VALUES
        }

        self.question_hits = {
            k: 0
            for k in TOP_K_VALUES
        }

        self.complete_questions = {
            k: 0
            for k in TOP_K_VALUES
        }


# ============================================================
# Gold evidence helpers
# ============================================================

def get_gold_evidence(
    example,
) -> Set[EvidenceKey]:
    """
    Convert the HotpotQA supporting facts for one question
    into exact sentence-level evidence keys:

        (document_id, sentence_id)

    Negative sentence IDs and missing document IDs are excluded.

    NOTE:
    For the later full held-out evaluation, gold validity will
    be checked against the authoritative corpus/evidence mapping.
    """

    gold: Set[EvidenceKey] = set()

    for fact in example.supporting_facts:

        if fact.document_id is None:
            continue

        if fact.sentence_id < 0:
            continue

        gold.add(
            (
                fact.document_id,
                fact.sentence_id,
            )
        )

    return gold


# ============================================================
# Retrieved evidence helpers
# ============================================================

def get_retrieved_evidence(
    results,
) -> List[Set[EvidenceKey]]:
    """
    Convert each retrieved chunk into the sentence-level
    evidence identities represented by that chunk.

    A sentence-aware chunk can contain multiple HotpotQA
    sentence IDs.
    """

    ranked_evidence: List[
        Set[EvidenceKey]
    ] = []

    for result in results:

        chunk = result.chunk

        sentence_ids = (
            chunk.metadata.get(
                "sentence_ids",
                [],
            )
        )

        evidence = {
            (
                chunk.document_id,
                sentence_id,
            )
            for sentence_id
            in sentence_ids
        }

        ranked_evidence.append(
            evidence
        )

    return ranked_evidence


def evidence_up_to_rank(
    ranked_evidence: List[
        Set[EvidenceKey]
    ],
    k: int,
) -> Set[EvidenceKey]:
    """
    Return the union of all sentence-level evidence represented
    by the first k retrieved chunks.
    """

    combined: Set[
        EvidenceKey
    ] = set()

    for evidence in ranked_evidence[:k]:
        combined.update(
            evidence
        )

    return combined


def first_relevant_rank(
    ranked_evidence: List[
        Set[EvidenceKey]
    ],
    gold: Set[EvidenceKey],
) -> int | None:
    """
    Return the rank of the first retrieved chunk containing
    at least one gold supporting fact.
    """

    for rank, evidence in enumerate(
        ranked_evidence,
        start=1,
    ):

        if evidence & gold:
            return rank

    return None


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    print("=" * 90)

    print(
        "PHASE 6.3 — SMALL-CORPUS "
        "DENSE RETRIEVAL EVALUATION"
    )

    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load HotpotQA
    # --------------------------------------------------------

    print(
        "\nLoading HotpotQA..."
    )

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_split = dataset["train"]

    # --------------------------------------------------------
    # 2. Build development corpus
    # --------------------------------------------------------

    print(
        f"Building corpus from first "
        f"{NUM_EXAMPLES} training examples..."
    )

    corpus = build_hotpotqa_corpus(
        train_split,
        max_examples=NUM_EXAMPLES,
    )

    print(
        f"QA examples: "
        f"{len(corpus.examples):,}"
    )

    print(
        f"Unique documents: "
        f"{len(corpus.documents):,}"
    )

    # --------------------------------------------------------
    # 3. Sentence-aware chunking
    # --------------------------------------------------------

    print(
        "\nSentence-aware chunking..."
    )

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
        f"Chunks: "
        f"{len(chunks):,}"
    )

    # --------------------------------------------------------
    # 4. Build retrieval representations
    # --------------------------------------------------------

    print(
        "\nPreparing retrieval representations..."
    )

    retrieval_texts = [
        format_chunk_for_retrieval(
            chunk
        )
        for chunk in chunks
    ]

    print(
        f"Retrieval texts: "
        f"{len(retrieval_texts):,}"
    )

    print(
        "\nSample retrieval representation:"
    )

    print(
        retrieval_texts[0][:500]
    )

    # --------------------------------------------------------
    # 5. Load dense embedding model
    # --------------------------------------------------------

    print(
        "\nLoading dense embedder..."
    )

    embedding_config = (
        DenseEmbeddingConfig(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            batch_size=32,
            normalize_embeddings=True,
        )
    )

    embedder = DenseEmbedder(
        embedding_config
    )

    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    # --------------------------------------------------------
    # 6. Embed corpus chunks ONCE
    # --------------------------------------------------------

    print(
        f"\nEmbedding "
        f"{len(retrieval_texts):,} "
        f"corpus chunks..."
    )

    chunk_embeddings = (
        embedder.encode_documents(
            retrieval_texts,
            show_progress_bar=True,
        )
    )

    print(
        f"Embedding matrix: "
        f"{chunk_embeddings.shape}"
    )

    # --------------------------------------------------------
    # 7. Build exact FAISS index
    # --------------------------------------------------------

    print(
        "\nBuilding exact FAISS "
        "IndexFlatIP retriever..."
    )

    retriever = DenseRetriever(
        chunks=chunks,
        embeddings=chunk_embeddings,
    )

    print(
        f"Indexed chunks: "
        f"{retriever.size:,}"
    )

    print(
        f"FAISS dimension: "
        f"{retriever.embedding_dimension}"
    )

    if retriever.size != len(chunks):
        raise RuntimeError(
            "FAISS index size does not "
            "match chunk count."
        )

    # --------------------------------------------------------
    # 8. Initialize evaluation
    # --------------------------------------------------------

    totals = EvaluationTotals()

    failure_examples = []

    print(
        "\nEvaluating questions...\n"
    )

    # --------------------------------------------------------
    # 9. Evaluate every question
    # --------------------------------------------------------

    for question_index, example in enumerate(
        corpus.examples,
        start=1,
    ):

        gold = get_gold_evidence(
            example
        )

        # No usable gold evidence means this question
        # cannot contribute to evidence-retrieval metrics.
        if not gold:
            continue

        # ---------------------------------------------
        # Dense retrieval
        # ---------------------------------------------

        results = retriever.search(
            query=example.question,
            embedder=embedder,
            top_k=max(TOP_K_VALUES),
        )

        ranked_evidence = (
            get_retrieved_evidence(
                results
            )
        )

        totals.questions += 1

        totals.valid_gold_facts += (
            len(gold)
        )

        # ---------------------------------------------
        # MRR
        # ---------------------------------------------

        first_rank = first_relevant_rank(
            ranked_evidence,
            gold,
        )

        if first_rank is not None:
            totals.reciprocal_rank_sum += (
                1.0 / first_rank
            )

        # ---------------------------------------------
        # Recall / hit / complete coverage
        # ---------------------------------------------

        for k in TOP_K_VALUES:

            retrieved = evidence_up_to_rank(
                ranked_evidence,
                k,
            )

            matched = (
                gold
                & retrieved
            )

            # Number of individual gold facts retrieved.
            totals.evidence_found[k] += (
                len(matched)
            )

            # At least one gold fact retrieved.
            if matched:
                totals.question_hits[k] += 1

            # Every required gold fact retrieved.
            if gold.issubset(
                retrieved
            ):
                totals.complete_questions[k] += 1

        # ---------------------------------------------
        # Capture incomplete-evidence cases
        # ---------------------------------------------

        retrieved_at_10 = (
            evidence_up_to_rank(
                ranked_evidence,
                max(TOP_K_VALUES),
            )
        )

        if not gold.issubset(
            retrieved_at_10
        ):

            missing = (
                gold
                - retrieved_at_10
            )

            failure_examples.append(
                {
                    "question": (
                        example.question
                    ),
                    "gold_count": (
                        len(gold)
                    ),
                    "found_count": (
                        len(
                            gold
                            & retrieved_at_10
                        )
                    ),
                    "missing": missing,
                    "results": results,
                }
            )

        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        if (
            question_index % 10 == 0
            or question_index
            == len(corpus.examples)
        ):
            print(
                f"Processed "
                f"{question_index}/"
                f"{len(corpus.examples)} "
                f"questions"
            )

    # ========================================================
    # 10. Aggregate results
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"\nEvaluated questions: "
        f"{totals.questions:,}"
    )

    print(
        f"Valid gold evidence facts: "
        f"{totals.valid_gold_facts:,}"
    )

    for k in TOP_K_VALUES:

        if totals.valid_gold_facts:
            evidence_recall = (
                totals.evidence_found[k]
                / totals.valid_gold_facts
            )
        else:
            evidence_recall = 0.0

        if totals.questions:
            question_hit_rate = (
                totals.question_hits[k]
                / totals.questions
            )

            complete_rate = (
                totals.complete_questions[k]
                / totals.questions
            )
        else:
            question_hit_rate = 0.0
            complete_rate = 0.0

        print(
            f"\n--- K = {k} ---"
        )

        print(
            f"Evidence Recall@{k}: "
            f"{evidence_recall:.4f} "
            f"("
            f"{totals.evidence_found[k]}/"
            f"{totals.valid_gold_facts}"
            f")"
        )

        print(
            f"Question Hit@{k}: "
            f"{question_hit_rate:.4f} "
            f"("
            f"{totals.question_hits[k]}/"
            f"{totals.questions}"
            f")"
        )

        print(
            f"Complete Evidence@{k}: "
            f"{complete_rate:.4f} "
            f"("
            f"{totals.complete_questions[k]}/"
            f"{totals.questions}"
            f")"
        )

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    if totals.questions:
        mrr = (
            totals.reciprocal_rank_sum
            / totals.questions
        )
    else:
        mrr = 0.0

    print(
        f"\nMRR@{max(TOP_K_VALUES)}: "
        f"{mrr:.4f}"
    )

    # --------------------------------------------------------
    # Incomplete evidence summary
    # --------------------------------------------------------

    print(
        f"\nQuestions with incomplete "
        f"evidence@{max(TOP_K_VALUES)}: "
        f"{len(failure_examples)}/"
        f"{totals.questions}"
    )

    # ========================================================
    # 11. Inspect sample failures
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SAMPLE INCOMPLETE-EVIDENCE CASES"
    )

    print(
        "=" * 90
    )

    if not failure_examples:

        print(
            "\nNo incomplete-evidence cases "
            "were found in this development slice."
        )

    else:

        for failure_index, failure in enumerate(
            failure_examples[:5],
            start=1,
        ):

            print(
                f"\n--- Failure "
                f"{failure_index} ---"
            )

            print(
                f"Question: "
                f"{failure['question']}"
            )

            print(
                f"Gold facts: "
                f"{failure['gold_count']}"
            )

            print(
                f"Gold facts found@"
                f"{max(TOP_K_VALUES)}: "
                f"{failure['found_count']}"
            )

            print(
                f"Missing evidence: "
                f"{sorted(failure['missing'])}"
            )

            print(
                "\nTop retrieved chunks:"
            )

            for rank, result in enumerate(
                failure["results"][:5],
                start=1,
            ):

                chunk = result.chunk

                title = (
                    chunk.metadata.get(
                        "title",
                        "<unknown>",
                    )
                )

                sentence_ids = (
                    chunk.metadata.get(
                        "sentence_ids",
                        [],
                    )
                )

                preview = (
                    chunk.text[:180]
                    .replace(
                        "\n",
                        " ",
                    )
                )

                print(
                    f"\n  Rank {rank}"
                )

                print(
                    f"    Title: "
                    f"{title!r}"
                )

                print(
                    f"    Score: "
                    f"{result.score:.4f}"
                )

                print(
                    f"    Document ID: "
                    f"{chunk.document_id}"
                )

                print(
                    f"    Sentence IDs: "
                    f"{sentence_ids}"
                )

                print(
                    f"    Text: "
                    f"{preview}"
                )

    # ========================================================
    # 12. Final note
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PHASE 6.3 EVALUATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "\nNOTE:"
    )

    print(
        "These are development-slice retrieval "
        "results over the first 100 HotpotQA "
        "training examples."
    )

    print(
        "They are intended for pipeline validation "
        "and failure analysis, not final held-out "
        "benchmark reporting."
    )


if __name__ == "__main__":
    main()