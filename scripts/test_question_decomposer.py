
"""Phase 11.4G-1 conservative question decomposer tests."""

from src.recovery.question_decomposer import (
    ConservativeQuestionDecomposer,
    DecompositionStatus,
)


DECOMPOSER = ConservativeQuestionDecomposer()


def test_standard_two_hop():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas attend?"
    )

    assert result.is_usable
    assert len(result.requirements.relationships) == 2
    assert result.requirements.answer_variable == "?university"


def test_correct_title():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas attend?"
    )

    assert (
        result.requirements.relationships[0].subject
        == "The Silver Atlas"
    )


def test_shared_author_variable():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas attend?"
    )

    first, second = result.requirements.relationships

    assert first.object == "?author"
    assert second.subject == "?author"


def test_predicates():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas attend?"
    )

    first, second = result.requirements.relationships

    assert first.predicate == "authored_by"
    assert second.predicate == "attended"


def test_case_insensitive_template():
    result = DECOMPOSER.decompose(
        "which UNIVERSITY did the AUTHOR of The Silver Atlas ATTEND?"
    )

    assert result.is_usable


def test_extra_whitespace():
    result = DECOMPOSER.decompose(
        " Which   university did the author of "
        "The Silver Atlas attend? "
    )

    assert result.is_usable


def test_quoted_title():
    result = DECOMPOSER.decompose(
        'Which university did the author of "The Silver Atlas" attend?'
    )

    assert result.is_usable
    assert (
        result.requirements.relationships[0].subject
        == "The Silver Atlas"
    )


def test_unsupported_paraphrase():
    result = DECOMPOSER.decompose(
        "Where did the writer of The Silver Atlas study?"
    )

    assert result.status == DecompositionStatus.UNSUPPORTED
    assert result.requirements is None


def test_unsupported_three_hop():
    result = DECOMPOSER.decompose(
        "Which city is the author's university located in?"
    )

    assert result.status == DecompositionStatus.UNSUPPORTED


def test_unsupported_single_hop():
    result = DECOMPOSER.decompose(
        "Who wrote The Silver Atlas?"
    )

    assert result.status == DecompositionStatus.UNSUPPORTED


def test_unsupported_other_relation():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas teach at?"
    )

    assert result.status == DecompositionStatus.UNSUPPORTED


def test_missing_question_mark():
    result = DECOMPOSER.decompose(
        "Which university did the author of The Silver Atlas attend"
    )

    assert result.status == DecompositionStatus.UNSUPPORTED


def test_empty_question():
    try:
        DECOMPOSER.decompose("  ")
    except ValueError:
        return

    raise AssertionError("Expected ValueError.")


def test_invalid_question_type():
    try:
        DECOMPOSER.decompose(None)
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_standard_two_hop,
        test_correct_title,
        test_shared_author_variable,
        test_predicates,
        test_case_insensitive_template,
        test_extra_whitespace,
        test_quoted_title,
        test_unsupported_paraphrase,
        test_unsupported_three_hop,
        test_unsupported_single_hop,
        test_unsupported_other_relation,
        test_missing_question_mark,
        test_empty_question,
        test_invalid_question_type,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4G-1 QUESTION DECOMPOSER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
