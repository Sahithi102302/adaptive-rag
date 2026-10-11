from __future__ import annotations

from dataclasses import dataclass

from src.chunking.chunk import Chunk
from src.recovery.evidence_selection import (
    EvidencePreservingSelector,
    EvidenceSelectionConfig,
)
from src.retrieval.reranker import RerankedRetrievalResult


FINAL_K = 10


@dataclass(frozen=True)
class Scenario:
    name: str
    first: tuple[RerankedRetrievalResult, ...]
    second: tuple[RerankedRetrievalResult, ...]
    gold_by_chunk: dict[str, frozenset[str]]
    required_facts: frozenset[str]


def make_result(chunk_id: str, rank: int) -> RerankedRetrievalResult:
    text = f"Synthetic evidence passage {chunk_id}."

    chunk = Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        metadata={"title": chunk_id},
    )

    return RerankedRetrievalResult(
        rank=rank,
        score=float(10 - rank),
        chunk=chunk,
    )


def make_results(ids: list[str]) -> tuple[RerankedRetrievalResult, ...]:
    return tuple(
        make_result(chunk_id, rank)
        for rank, chunk_id in enumerate(ids, start=1)
    )


def first_hop_only(scenario: Scenario) -> list[str]:
    return [result.chunk.id for result in scenario.first[:FINAL_K]]


def naive_rank_union(scenario: Scenario) -> list[str]:
    """
    Deterministic baseline:
    prioritize the first-hop ranking, then append unseen
    second-hop chunks until the final budget is filled.

    This is NOT a cross-encoder reranking baseline.
    """
    selected = []
    seen = set()

    for result in (*scenario.first, *scenario.second):
        chunk_id = result.chunk.id

        if chunk_id not in seen and len(selected) < FINAL_K:
            selected.append(chunk_id)
            seen.add(chunk_id)

    return selected


def hop_balanced(scenario: Scenario) -> list[str]:
    selector = EvidencePreservingSelector(
        EvidenceSelectionConfig(
            final_k=FINAL_K,
            reserved_first_hop=5,
            reserved_second_hop=5,
        )
    )

    selection = selector.select(
        original_results=scenario.first,
        recovered_results=scenario.second,
    )

    return [
        item.chunk_id
        for item in selection.selected_results
    ]


def evidence_for(
    selected_ids: list[str],
    scenario: Scenario,
) -> set[str]:
    evidence = set()

    for chunk_id in selected_ids:
        evidence.update(
            scenario.gold_by_chunk.get(chunk_id, frozenset())
        )

    return evidence


def evaluate(
    selected_ids: list[str],
    scenario: Scenario,
) -> dict:
    first_ids = first_hop_only(scenario)
    first_evidence = evidence_for(first_ids, scenario)
    selected_evidence = evidence_for(selected_ids, scenario)

    initially_missing = scenario.required_facts - first_evidence
    recovered = initially_missing & selected_evidence
    lost = (first_evidence & scenario.required_facts) - selected_evidence

    return {
        "complete": scenario.required_facts.issubset(selected_evidence),
        "recovered": len(recovered),
        "lost": len(lost),
        "selected": selected_ids,
    }


def build_scenarios() -> list[Scenario]:
    # Scenario 1:
    # Five first-hop chunks contain necessary facts.
    # The second hop supplies the missing sixth fact.
    repair = Scenario(
        name="Successful repair",
        first=make_results(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        ),
        second=make_results(
            ["K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
        ),
        gold_by_chunk={
            "A": frozenset({"f1"}),
            "B": frozenset({"f2"}),
            "C": frozenset({"f3"}),
            "D": frozenset({"f4"}),
            "E": frozenset({"f5"}),
            "K": frozenset({"f6"}),
        },
        required_facts=frozenset(
            {"f1", "f2", "f3", "f4", "f5", "f6"}
        ),
    )

    # Scenario 2:
    # A necessary original fact occurs at rank 8.
    # The balanced policy drops it while adding new evidence.
    harm = Scenario(
        name="Recovery harm",
        first=make_results(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        ),
        second=make_results(
            ["K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
        ),
        gold_by_chunk={
            "H": frozenset({"f1"}),
            "K": frozenset({"f2"}),
        },
        required_facts=frozenset({"f1", "f2"}),
    )

    # Scenario 3:
    # Some chunks appear in both hops.
    overlap = Scenario(
        name="Overlapping retrieval",
        first=make_results(
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        ),
        second=make_results(
            ["C", "D", "K", "L", "M", "N", "O", "P", "Q", "R"]
        ),
        gold_by_chunk={
            "C": frozenset({"f1"}),
            "D": frozenset({"f2"}),
            "K": frozenset({"f3"}),
        },
        required_facts=frozenset({"f1", "f2", "f3"}),
    )

    return [repair, harm, overlap]


def main() -> None:
    print("PHASE 10.4B — SYNTHETIC SELECTION POLICY COMPARISON")

    scenarios = build_scenarios()

    policies = {
        "First-hop only": first_hop_only,
        "Naive rank union": naive_rank_union,
        "Hop-balanced": hop_balanced,
    }

    for scenario in scenarios:
        print("\n" + "=" * 80)
        print(scenario.name)
        print("=" * 80)

        for name, policy in policies.items():
            selected_ids = policy(scenario)
            metrics = evaluate(selected_ids, scenario)

            assert len(selected_ids) <= FINAL_K
            assert len(selected_ids) == len(set(selected_ids))

            print(f"\n{name}")
            print(f"  Selected: {selected_ids}")
            print(f"  Complete evidence: {metrics['complete']}")
            print(f"  Newly recovered facts: {metrics['recovered']}")
            print(f"  Lost original facts: {metrics['lost']}")

    print("\nPHASE 10.4B SYNTHETIC COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()