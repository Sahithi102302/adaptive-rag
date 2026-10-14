
"""Phase 11.4G-7 semantic consistency gate tests."""

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
)
from src.recovery.decomposition_semantic_gate import (
    DecompositionSemanticGate,
)


VALIDATOR = DecompositionProposalValidator()
GATE = DecompositionSemanticGate()


def proposal(
    title="The Silver Atlas",
    author_var="?author",
    answer_var="?university",
):
    return {
        "relationships": [
            {
                "subject": title,
                "predicate": "authored_by",
                "object": author_var,
            },
            {
                "subject": author_var,
                "predicate": "attended",
                "object": answer_var,
            },
        ],
        "answer_variable": answer_var,
    }


def accepted(question, candidate):
    validation = VALIDATOR.validate(question, candidate)
    assert validation.is_valid, validation.explanation
    return GATE.check(
        question,
        validation.requirements,
    ).accepted


def test_exact_template():
    assert accepted(
        "Which university did the author of The Silver Atlas attend?",
        proposal(),
    )


def test_writer_paraphrase():
    assert accepted(
        "Where did the writer of The Silver Atlas study?",
        proposal(author_var="?writer"),
    )


def test_what_university():
    assert accepted(
        "What university did the author of The Silver Atlas attend?",
        proposal(),
    )


def test_college():
    assert accepted(
        "Which college did the author of The Silver Atlas attend?",
        proposal(answer_var="?college"),
    )


def test_at_which_university():
    assert accepted(
        "At which university did the writer of The Silver Atlas study?",
        proposal(author_var="?writer"),
    )


def test_missing_question_mark():
    assert accepted(
        "Which university did the author of The Silver Atlas attend",
        proposal(),
    )


def test_quoted_title():
    assert accepted(
        'Which university did the author of "The Silver Atlas" attend?',
        proposal(),
    )


def test_different_book():
    assert accepted(
        "Which university did the author of The Hidden Garden attend?",
        proposal(title="The Hidden Garden"),
    )


def test_wrong_book_rejected():
    assert not accepted(
        "Which university did the author of The Silver Atlas attend?",
        proposal(title="Another Book"),
    )


def test_taught_at_rejected():
    assert not accepted(
        "Which university did the author of The Silver Atlas teach at?",
        proposal(),
    )


def test_illustrator_rejected():
    assert not accepted(
        "Which university did the illustrator of The Silver Atlas attend?",
        proposal(author_var="?illustrator"),
    )


def test_editor_rejected():
    assert not accepted(
        "Which university did the editor of The Silver Atlas attend?",
        proposal(author_var="?editor"),
    )


def test_single_hop_scope_rejected():
    candidate = {
        "relationships": [
            {
                "subject": "The Silver Atlas",
                "predicate": "authored_by",
                "object": "?author",
            }
        ],
        "answer_variable": "?author",
    }

    assert not accepted(
        "Who wrote The Silver Atlas?",
        candidate,
    )


def test_city_question_rejected():
    assert not accepted(
        "Which city is the author's university located in?",
        proposal(),
    )


def test_extra_graduation_request_rejected():
    assert not accepted(
        "Which university did the author of The Silver Atlas attend, "
        "and when did they graduate?",
        proposal(),
    )


def test_wrong_answer_target_rejected():
    candidate = proposal()
    candidate["answer_variable"] = "?author"

    assert not accepted(
        "Which university did the author of The Silver Atlas attend?",
        candidate,
    )


def test_wrong_relation_direction_rejected():
    candidate = {
        "relationships": [
            {
                "subject": "?author",
                "predicate": "authored_by",
                "object": "The Silver Atlas",
            },
            {
                "subject": "?author",
                "predicate": "attended",
                "object": "?university",
            },
        ],
        "answer_variable": "?university",
    }

    assert not accepted(
        "Which university did the author of The Silver Atlas attend?",
        candidate,
    )


def main():
    tests = [
        test_exact_template,
        test_writer_paraphrase,
        test_what_university,
        test_college,
        test_at_which_university,
        test_missing_question_mark,
        test_quoted_title,
        test_different_book,
        test_wrong_book_rejected,
        test_taught_at_rejected,
        test_illustrator_rejected,
        test_editor_rejected,
        test_single_hop_scope_rejected,
        test_city_question_rejected,
        test_extra_graduation_request_rejected,
        test_wrong_answer_target_rejected,
        test_wrong_relation_direction_rejected,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4G-7 SEMANTIC GATE TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
