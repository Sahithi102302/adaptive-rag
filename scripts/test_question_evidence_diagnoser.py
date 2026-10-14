
"""Phase 11.4G-2 question-to-evidence diagnosis tests."""

from src.recovery.question_evidence_diagnoser import (
    QuestionEvidenceDiagnoser,
    DiagnosisStatus,
)


DIAGNOSER = QuestionEvidenceDiagnoser()

QUESTION = (
    "Which university did the author of "
    "The Silver Atlas attend?"
)

BOOK = "The Silver Atlas was written by Elena Marsh."
EDUCATION = "Elena Marsh attended Northbridge University."
NEGATIVE = "Elena Marsh never attended Northbridge University."


def test_complete_two_hop():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {"book": BOOK, "education": EDUCATION},
    )

    assert result.status == DiagnosisStatus.UNIQUE_CANDIDATE
    assert result.clean_candidate_answers == (
        "Northbridge University",
    )
    assert result.coverage_report is not None


def test_evidence_citations():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {"book": BOOK, "education": EDUCATION},
    )

    paths = result.coverage_report.path_summaries

    assert len(paths) == 1
    assert paths[0].supporting_chunk_ids == (
        "book", "education"
    )


def test_missing_second_hop():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {"book": BOOK},
    )

    assert result.status == DiagnosisStatus.INCOMPLETE
    assert result.clean_candidate_answers == ()


def test_missing_first_hop():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {"education": EDUCATION},
    )

    assert result.status == DiagnosisStatus.INCOMPLETE


def test_disconnected_evidence():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {
            "book": BOOK,
            "education": (
                "Daniel Reed attended Lakeside College."
            ),
        },
    )

    assert result.status == DiagnosisStatus.INCOMPLETE
    assert result.coverage_report.contradictions == ()


def test_explicit_contradiction():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {
            "book": BOOK,
            "education": EDUCATION,
            "negative": NEGATIVE,
        },
    )

    assert result.status == DiagnosisStatus.CONFLICTING
    assert result.clean_candidate_answers == ()
    assert len(
        result.coverage_report.contradictions
    ) == 1


def test_same_chunk_contradiction():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {
            "book": BOOK,
            "education": EDUCATION + " " + NEGATIVE,
        },
    )

    assert result.status == DiagnosisStatus.CONFLICTING


def test_irrelevant_distractor():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {
            "book": BOOK,
            "education": EDUCATION,
            "distractor": (
                "Daniel Reed attended Lakeside College."
            ),
        },
    )

    assert result.status == DiagnosisStatus.UNIQUE_CANDIDATE
    assert result.clean_candidate_answers == (
        "Northbridge University",
    )


def test_ambiguous_answer():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {
            "book": BOOK,
            "a": EDUCATION,
            "b": "Elena Marsh attended Lakeside College.",
        },
    )

    assert result.status == DiagnosisStatus.AMBIGUOUS
    assert len(result.clean_candidate_answers) == 2


def test_unsupported_question():
    result = DIAGNOSER.diagnose(
        "Where did the writer of The Silver Atlas study?",
        {"book": BOOK, "education": EDUCATION},
    )

    assert result.status == DiagnosisStatus.UNSUPPORTED_QUESTION
    assert result.coverage_report is None
    assert result.clean_candidate_answers == ()


def test_unsupported_three_hop():
    result = DIAGNOSER.diagnose(
        "Which city is the author's university located in?",
        {"book": BOOK, "education": EDUCATION},
    )

    assert result.status == DiagnosisStatus.UNSUPPORTED_QUESTION


def test_empty_evidence():
    result = DIAGNOSER.diagnose(QUESTION, {})

    assert result.status == DiagnosisStatus.INCOMPLETE


def test_quoted_book_title():
    result = DIAGNOSER.diagnose(
        'Which university did the author of "The Silver Atlas" attend?',
        {"book": BOOK, "education": EDUCATION},
    )

    assert result.status == DiagnosisStatus.UNIQUE_CANDIDATE


def test_no_automatic_acceptance():
    result = DIAGNOSER.diagnose(
        QUESTION,
        {"book": BOOK, "education": EDUCATION},
    )

    assert result.eligible_for_automatic_acceptance is False
    assert (
        result.coverage_report.eligible_for_automatic_acceptance
        is False
    )


def test_invalid_evidence_type():
    try:
        DIAGNOSER.diagnose(QUESTION, ["not", "a", "mapping"])
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def test_invalid_evidence_passage():
    try:
        DIAGNOSER.diagnose(
            QUESTION,
            {"book": 123},
        )
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def test_invalid_question():
    try:
        DIAGNOSER.diagnose(None, {})
    except TypeError:
        return

    raise AssertionError("Expected TypeError.")


def main():
    tests = [
        test_complete_two_hop,
        test_evidence_citations,
        test_missing_second_hop,
        test_missing_first_hop,
        test_disconnected_evidence,
        test_explicit_contradiction,
        test_same_chunk_contradiction,
        test_irrelevant_distractor,
        test_ambiguous_answer,
        test_unsupported_question,
        test_unsupported_three_hop,
        test_empty_evidence,
        test_quoted_book_title,
        test_no_automatic_acceptance,
        test_invalid_evidence_type,
        test_invalid_evidence_passage,
        test_invalid_question,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4G-2 QUESTION EVIDENCE "
        "DIAGNOSER TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
