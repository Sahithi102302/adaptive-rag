
"""Phase 11.3A tests for offline verification scoring."""

from types import SimpleNamespace

from src.recovery.offline_verification import (
    evaluate_offline_verification,
)


def evidence(chunk_id, document_id, text, facts):
    return SimpleNamespace(
        chunk=SimpleNamespace(
            id=chunk_id,
            document_id=document_id,
            text=text,
        ),
        facts=set(facts),
    )


def extract_facts(results):
    facts = set()
    for result in results:
        facts.update(result.facts)
    return facts


def evaluate(original, candidate, gold):
    return evaluate_offline_verification(
        query="Which city hosted the Olympics?",
        original_results=original,
        candidate_results=candidate,
        gold_evidence=set(gold),
        evidence_extractor=extract_facts,
    )


def test_successful_repair():
    original = [
        evidence("a", "doc1", "Olympics event.", {("doc1", 0)})
    ]
    candidate = [
        original[0],
        evidence("b", "doc2", "London hosted the Olympics.", {("doc2", 1)}),
    ]

    result = evaluate(
        original, candidate, {("doc1", 0), ("doc2", 1)}
    )

    assert result.repaired
    assert result.candidate_complete
    assert result.recovered_count == 1
    assert result.lost_count == 0


def test_recovery_harm():
    original = [
        evidence("a", "doc1", "Olympics event.", {("doc1", 0)})
    ]
    candidate = [
        evidence("b", "doc2", "London hosted the Olympics.", {("doc2", 1)})
    ]

    result = evaluate(
        original, candidate, {("doc1", 0), ("doc2", 1)}
    )

    assert result.harmed
    assert result.lost_count == 1
    assert result.recovered_count == 1
    assert not result.repaired


def test_complete_to_incomplete():
    original = [
        evidence("a", "doc1", "Olympics event.", {("doc1", 0)}),
        evidence("b", "doc2", "London hosted the Olympics.", {("doc2", 1)}),
    ]
    candidate = [original[0]]

    result = evaluate(
        original, candidate, {("doc1", 0), ("doc2", 1)}
    )

    assert result.originally_complete
    assert result.complete_to_incomplete
    assert result.harmed


def test_lexical_gain_not_proof_of_repair():
    original = [
        evidence("a", "doc1", "Olympics event.", {("doc1", 0)})
    ]
    candidate = [
        original[0],
        evidence("b", "doc2", "Which city hosted the Olympics?", set()),
    ]

    result = evaluate(
        original, candidate, {("doc1", 0), ("doc2", 1)}
    )

    assert result.comparison.coverage_delta > 0
    assert not result.repaired
    assert result.recovered_count == 0


def test_empty_gold_rejected():
    original = [
        evidence("a", "doc1", "Olympics event.", set())
    ]

    try:
        evaluate(original, original, set())
    except ValueError:
        return

    raise AssertionError("Expected ValueError for empty gold evidence.")


def main():
    tests = [
        test_successful_repair,
        test_recovery_harm,
        test_complete_to_incomplete,
        test_lexical_gain_not_proof_of_repair,
        test_empty_gold_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.3A OFFLINE VERIFICATION TESTS PASSED")


if __name__ == "__main__":
    main()
