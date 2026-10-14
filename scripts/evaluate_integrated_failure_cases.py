
"""Phase 11.4F-7: integrated failure-case evaluation.

Runs five fixed synthetic evidence scenarios using manually
declared question requirements. This is a diagnostic evaluation,
not a HotpotQA benchmark or an end-to-end LLM evaluation.
"""

from dataclasses import dataclass

from src.recovery.integrated_coverage_reporter import (
    IntegratedCoverageReporter,
    IntegratedDecision,
)
from src.recovery.question_requirements import (
    QuestionRequirements,
    RequiredRelationship,
)


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    description: str
    evidence: dict[str, str]
    expected: IntegratedDecision
    expected_answer: str | None = None


def build_requirements() -> QuestionRequirements:
    return QuestionRequirements(
        question=(
            "Which university did the author of "
            "The Silver Atlas attend?"
        ),
        relationships=(
            RequiredRelationship(
                relationship_id="author",
                subject="The Silver Atlas",
                predicate="authored_by",
                object="?author",
            ),
            RequiredRelationship(
                relationship_id="education",
                subject="?author",
                predicate="attended",
                object="?university",
            ),
        ),
        answer_variable="?university",
    )


BOOK = "The Silver Atlas was written by Elena Marsh."
EDUCATION = "Elena Marsh attended Northbridge University."
NEGATIVE = "Elena Marsh never attended Northbridge University."


CASES = (
    FailureCase(
        case_id="A",
        description="Complete two-hop evidence",
        evidence={
            "book": BOOK,
            "education": EDUCATION,
        },
        expected=IntegratedDecision.UNIQUE_CANDIDATE,
        expected_answer="Northbridge University",
    ),
    FailureCase(
        case_id="B",
        description="Missing second hop",
        evidence={"book": BOOK},
        expected=IntegratedDecision.INCOMPLETE,
    ),
    FailureCase(
        case_id="C",
        description="Missing first hop",
        evidence={"education": EDUCATION},
        expected=IntegratedDecision.INCOMPLETE,
    ),
    FailureCase(
        case_id="D",
        description="Disconnected evidence",
        evidence={
            "book": BOOK,
            "education": (
                "Daniel Reed attended Lakeside College."
            ),
        },
        expected=IntegratedDecision.INCOMPLETE,
    ),
    FailureCase(
        case_id="E",
        description="Explicit contradiction",
        evidence={
            "book": BOOK,
            "education": EDUCATION,
            "negative": NEGATIVE,
        },
        expected=IntegratedDecision.CONFLICTING,
    ),
)


def main() -> None:
    reporter = IntegratedCoverageReporter()
    requirements = build_requirements()

    correct = 0
    false_acceptances = 0
    false_rejections = 0

    for case in CASES:
        report = reporter.assess(
            requirements,
            case.evidence,
        )

        decision_matches = report.decision == case.expected

        answer_matches = (
            case.expected_answer is None
            or report.clean_candidate_answers
            == (case.expected_answer,)
        )

        passed = decision_matches and answer_matches

        if passed:
            correct += 1

        # Candidate-level diagnostic only; automatic
        # acceptance remains disabled in the reporter.
        if (
            case.expected != IntegratedDecision.UNIQUE_CANDIDATE
            and report.decision
            == IntegratedDecision.UNIQUE_CANDIDATE
        ):
            false_acceptances += 1

        if (
            case.expected == IntegratedDecision.UNIQUE_CANDIDATE
            and report.decision
            != IntegratedDecision.UNIQUE_CANDIDATE
        ):
            false_rejections += 1

        print("=" * 65)
        print(f"CASE {case.case_id}: {case.description}")
        print(f"Expected: {case.expected.value}")
        print(f"Observed: {report.decision.value}")
        print(f"Candidate answers: {report.candidate_answers}")
        print(
            "Clean candidate answers: "
            f"{report.clean_candidate_answers}"
        )
        print(
            "Explicit contradictions: "
            f"{len(report.contradictions)}"
        )
        print(
            "Independent coverage: "
            f"{report.independent_coverage.decision.value}"
        )
        print(
            "Automatic acceptance eligible: "
            f"{report.eligible_for_automatic_acceptance}"
        )

        for index, path in enumerate(
            report.path_summaries, start=1
        ):
            print(f"  Path {index}:")
            print(f"    Answer: {path.answer_value}")
            print(f"    Bindings: {path.bindings}")
            print(
                "    Evidence chunks: "
                f"{path.supporting_chunk_ids}"
            )
            print(
                "    Contradictions: "
                f"{len(path.contradictions)}"
            )

        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    total = len(CASES)

    print("\n" + "=" * 65)
    print("PHASE 11.4F-7 — SYNTHETIC FAILURE-CASE EVALUATION")
    print(f"Correct classifications: {correct}/{total}")
    print(f"False candidate acceptances: {false_acceptances}")
    print(f"False candidate rejections: {false_rejections}")
    print("Automatic acceptance: disabled")
    print("=" * 65)

    if correct != total:
        raise AssertionError(
            f"Failure-case evaluation: {correct}/{total} passed."
        )


if __name__ == "__main__":
    main()
