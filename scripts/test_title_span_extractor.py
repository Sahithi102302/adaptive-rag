
"""Phase 11.4G-12H-1: title-span extractor tests.

Tests recognized and unresolved title-boundary cases.
"""

from __future__ import annotations

from src.recovery.title_span_extractor import (
    ConservativeTitleSpanExtractor,
    TitleSpanStatus,
)


extractor = ConservativeTitleSpanExtractor()


def assert_title(question: str, expected: str) -> None:
    result = extractor.extract(question)

    assert result.status == TitleSpanStatus.FOUND, result
    assert result.title == expected, result

    assert result.start is not None
    assert result.end is not None

    assert question[result.start:result.end] == expected


def assert_unresolved(question: str) -> None:
    result = extractor.extract(question)

    assert result.status == TitleSpanStatus.UNRESOLVED, result
    assert result.title is None


def test_basic_title():
    assert_title(
        "Which university did the author of "
        "The Copper Horizon attend?",
        "The Copper Horizon",
    )


def test_title_with_negation_word():
    assert_title(
        "Which university did the author of "
        "The Never Ending Journey attend?",
        "The Never Ending Journey",
    )


def test_title_with_before():
    assert_title(
        "Which university did the author of "
        "Before the Sunrise attend?",
        "Before the Sunrise",
    )


def test_title_with_role_word():
    assert_title(
        "Which university did the author of "
        "The Editor's Notebook attend?",
        "The Editor's Notebook",
    )


def test_title_with_year():
    assert_title(
        "Which university did the author of "
        "Summer in 1998 attend?",
        "Summer in 1998",
    )


def test_constraint_outside_title():
    assert_title(
        "Which university did the author of "
        "The Copper Horizon attend in 1998?",
        "The Copper Horizon",
    )


def test_before_constraint_outside_title():
    assert_title(
        "Which university did the author of "
        "The Copper Horizon attend before 2005?",
        "The Copper Horizon",
    )


def test_title_with_relation_word():
    assert_title(
        "Which university did the author of "
        "The University That Hired Me attend?",
        "The University That Hired Me",
    )


def test_one_hop_question():
    assert_unresolved(
        "Who authored The Copper Horizon?"
    )


def test_missing_title():
    assert_unresolved(
        "Which university did the author attend?"
    )


def test_empty_question():
    assert_unresolved("")


def test_alternate_authorship_title():
    assert_title(
        "Where did the person who authored "
        "The Paper Constellation go to college?",
        "The Paper Constellation",
    )


def main() -> None:
    tests = [
        test_basic_title,
        test_title_with_negation_word,
        test_title_with_before,
        test_title_with_role_word,
        test_title_with_year,
        test_constraint_outside_title,
        test_before_constraint_outside_title,
        test_title_with_relation_word,
        test_one_hop_question,
        test_missing_title,
        test_empty_question,
        test_alternate_authorship_title,
    ]

    passed = 0

    for test in tests:
        test()
        passed += 1
        print(f"PASS: {test.__name__}")

    print()
    print("PHASE 11.4G-12H-1 TITLE EXTRACTION TESTS PASSED")
    print(f"Tests passed: {passed}/{len(tests)}")


if __name__ == "__main__":
    main()
