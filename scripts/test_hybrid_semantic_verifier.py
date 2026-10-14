
"""Phase 11.4G-12B: hybrid semantic router tests."""

from src.recovery.hybrid_semantic_verifier import (
    HybridDecision,
    HybridReason,
    HybridSemanticVerifier,
)


verifier = HybridSemanticVerifier()


def proposal(
    title="The Copper Horizon",
    author="?author",
    answer="?university",
):
    return {
        "relationships": [
            {
                "subject": title,
                "predicate": "authored_by",
                "object": author,
            },
            {
                "subject": author,
                "predicate": "attended",
                "object": answer,
            },
        ],
        "answer_variable": answer,
    }


def test_verified_grammar_accepts():
    result = verifier.verify(
        "Which university did the author of The Copper Horizon attend?",
        proposal(),
    )
    assert result.decision == HybridDecision.ACCEPT
    assert result.reason == HybridReason.VERIFIED_GRAMMAR


def test_unseen_paraphrase_abstains():
    result = verifier.verify(
        "Which university did the novelist behind The Copper Horizon attend?",
        proposal(),
    )
    assert result.decision == HybridDecision.ABSTAIN
    assert result.reason == HybridReason.OUT_OF_GRAMMAR


def test_unsupported_relation_does_not_accept():
    result = verifier.verify(
        "Which university employed the author of The Copper Horizon?",
        proposal(),
    )
    assert result.decision != HybridDecision.ACCEPT


def test_extra_constraint_does_not_accept():
    result = verifier.verify(
        "Which university did the author of The Copper Horizon attend first?",
        proposal(),
    )
    assert result.decision != HybridDecision.ACCEPT


def test_wrong_book_rejects():
    result = verifier.verify(
        "Which university did the author of The Copper Horizon attend?",
        proposal(title="The Paper Constellation"),
    )
    assert result.decision == HybridDecision.REJECT
    assert result.reason == HybridReason.IN_GRAMMAR_MISMATCH


def test_invalid_structure_rejects():
    candidate = proposal()
    candidate["relationships"][0]["predicate"] = "translated_by"

    result = verifier.verify(
        "Which university did the author of The Copper Horizon attend?",
        candidate,
    )
    assert result.decision == HybridDecision.REJECT
    assert result.reason == HybridReason.INVALID_STRUCTURE


def test_empty_question_rejects():
    result = verifier.verify(
        "",
        proposal(),
    )
    assert result.decision == HybridDecision.REJECT
    assert result.reason == HybridReason.INVALID_INPUT


def test_one_hop_out_of_scope_abstains():
    candidate = {
        "relationships": [
            {
                "subject": "The Copper Horizon",
                "predicate": "authored_by",
                "object": "?author",
            }
        ],
        "answer_variable": "?author",
    }

    result = verifier.verify(
        "Who wrote The Copper Horizon?",
        candidate,
    )
    assert result.decision == HybridDecision.ABSTAIN
    assert result.reason == HybridReason.OUT_OF_GRAMMAR


def main():
    tests = [
        test_verified_grammar_accepts,
        test_unseen_paraphrase_abstains,
        test_unsupported_relation_does_not_accept,
        test_extra_constraint_does_not_accept,
        test_wrong_book_rejects,
        test_invalid_structure_rejects,
        test_empty_question_rejects,
        test_one_hop_out_of_scope_abstains,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.4G-12 HYBRID ROUTER TESTS PASSED")
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
