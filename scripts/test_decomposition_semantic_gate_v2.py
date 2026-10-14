
"""Phase 11.4G-10: semantic gate v2 regression tests."""

from __future__ import annotations

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
)
from src.recovery.decomposition_semantic_gate import (
    DecompositionSemanticGate,
)


validator = DecompositionProposalValidator()
gate = DecompositionSemanticGate()


def proposal(
    title="The Amber Compass",
    author="?author",
    university="?university",
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
                "object": university,
            },
        ],
        "answer_variable": university,
    }


def evaluate(question, candidate):
    structural = validator.validate(question, candidate)

    assert structural.is_valid, structural.explanation

    return gate.check(
        question,
        structural.requirements,
    )


def test_original_template():
    assert evaluate(
        "Which university did the author of The Amber Compass attend?",
        proposal(),
    ).accepted


def test_passive_voice():
    assert evaluate(
        "Which university was attended by the author of The Amber Compass?",
        proposal(),
    ).accepted


def test_person_who_wrote():
    assert evaluate(
        "What college did the person who wrote The Glass Harbor go to?",
        proposal(
            title="The Glass Harbor",
            university="?college",
        ),
    ).accepted


def test_name_the_university():
    assert evaluate(
        "Name the university attended by the writer of The Blue Lantern.",
        proposal(
            title="The Blue Lantern",
            author="?writer",
        ),
    ).accepted


def test_inverted_question():
    assert evaluate(
        "The author of The Winter Orchard studied at which university?",
        proposal(title="The Winter Orchard"),
    ).accepted


def test_wrong_book():
    assert not evaluate(
        "Which university was attended by the author of The Amber Compass?",
        proposal(title="The Glass Harbor"),
    ).accepted


def test_worked_at():
    assert not evaluate(
        "Which university did the author of The Amber Compass work at?",
        proposal(),
    ).accepted


def test_graduated_from():
    assert not evaluate(
        "Which university did the author of The Amber Compass graduate from?",
        proposal(),
    ).accepted


def test_negation():
    assert not evaluate(
        "Which university did the author of The Amber Compass not attend?",
        proposal(),
    ).accepted


def test_editor():
    assert not evaluate(
        "Which university was attended by the editor of The Amber Compass?",
        proposal(),
    ).accepted


def test_translator():
    assert not evaluate(
        "Name the university attended by the translator of The Amber Compass.",
        proposal(),
    ).accepted


def test_compound_question():
    assert not evaluate(
        "Which university was attended by the author of "
        "The Amber Compass, and when?",
        proposal(),
    ).accepted


def test_wrong_answer_target():
    candidate = proposal()
    candidate["answer_variable"] = "?author"

    assert not evaluate(
        "Which university was attended by the author of The Amber Compass?",
        candidate,
    ).accepted


def test_wrong_relation_direction():
    candidate = {
        "relationships": [
            {
                "subject": "?author",
                "predicate": "authored_by",
                "object": "The Amber Compass",
            },
            {
                "subject": "?author",
                "predicate": "attended",
                "object": "?university",
            },
        ],
        "answer_variable": "?university",
    }

    assert not evaluate(
        "Which university was attended by the author of The Amber Compass?",
        candidate,
    ).accepted


def main():
    tests = [
        test_original_template,
        test_passive_voice,
        test_person_who_wrote,
        test_name_the_university,
        test_inverted_question,
        test_wrong_book,
        test_worked_at,
        test_graduated_from,
        test_negation,
        test_editor,
        test_translator,
        test_compound_question,
        test_wrong_answer_target,
        test_wrong_relation_direction,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nPHASE 11.4G-10 SEMANTIC GATE V2 TESTS PASSED")
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
