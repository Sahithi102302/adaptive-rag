
"""Phase 11.4G-9: held-out adversarial decomposition evaluation.

Evaluates the frozen Qwen + structural validator + semantic gate
pipeline on 24 new hand-authored questions.

Labels represent the restricted two-hop author/education task.
No retrieved evidence or gold supporting facts are provided.

Results are saved separately from the development benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.recovery.qwen_question_decomposer import (
    QwenQuestionDecomposer,
    QwenDecompositionStatus,
)


OUTPUT_PATH = Path(
    "outputs/phase11_4g9_heldout_decomposition_results.json"
)


@dataclass(frozen=True)
class HeldoutCase:
    case_id: str
    question: str
    expected_supported: bool
    expected_title: str | None
    category: str


CASES = (
    # New entities with familiar question structures.
    HeldoutCase(
        "H01",
        "Which university did the author of The Amber Compass attend?",
        True,
        "The Amber Compass",
        "new_entity",
    ),
    HeldoutCase(
        "H02",
        "Which college did the writer of The Glass Harbor attend?",
        True,
        "The Glass Harbor",
        "new_entity",
    ),
    HeldoutCase(
        "H03",
        "Where did the author of The Winter Orchard study?",
        True,
        "The Winter Orchard",
        "new_entity",
    ),
    HeldoutCase(
        "H04",
        "At which university did the author of The Blue Lantern study?",
        True,
        "The Blue Lantern",
        "new_entity",
    ),

    # Meaning-preserving paraphrases not covered by the
    # original semantic gate's exact question grammar.
    HeldoutCase(
        "H05",
        "Which university was attended by the author of The Amber Compass?",
        True,
        "The Amber Compass",
        "paraphrase",
    ),
    HeldoutCase(
        "H06",
        "What college did the person who wrote The Glass Harbor go to?",
        True,
        "The Glass Harbor",
        "paraphrase",
    ),
    HeldoutCase(
        "H07",
        "Name the university attended by the writer of The Blue Lantern.",
        True,
        "The Blue Lantern",
        "paraphrase",
    ),
    HeldoutCase(
        "H08",
        "The author of The Winter Orchard studied at which university?",
        True,
        "The Winter Orchard",
        "paraphrase",
    ),

    # Formatting and surface variations.
    HeldoutCase(
        "H09",
        'Which university did the author of "The Amber Compass" attend?',
        True,
        "The Amber Compass",
        "formatting",
    ),
    HeldoutCase(
        "H10",
        "WHICH UNIVERSITY DID THE AUTHOR OF THE GLASS HARBOR ATTEND?",
        True,
        "THE GLASS HARBOR",
        "formatting",
    ),
    HeldoutCase(
        "H11",
        "Which university did the author of The Blue Lantern attend",
        True,
        "The Blue Lantern",
        "formatting",
    ),
    HeldoutCase(
        "H12",
        "Where did the writer of The Winter Orchard study?",
        True,
        "The Winter Orchard",
        "formatting",
    ),

    # Semantic relation substitutions.
    HeldoutCase(
        "H13",
        "Which university did the author of The Amber Compass work at?",
        False,
        None,
        "relation_substitution",
    ),
    HeldoutCase(
        "H14",
        "Which university did the publisher of The Glass Harbor attend?",
        False,
        None,
        "relation_substitution",
    ),
    HeldoutCase(
        "H15",
        "Which university did the translator of The Blue Lantern attend?",
        False,
        None,
        "relation_substitution",
    ),
    HeldoutCase(
        "H16",
        "Which university did the author of The Winter Orchard graduate from?",
        False,
        None,
        "relation_substitution",
    ),

    # Negation, uncertainty, and misleading wording.
    HeldoutCase(
        "H17",
        "Which university did the author of The Amber Compass not attend?",
        False,
        None,
        "negation",
    ),
    HeldoutCase(
        "H18",
        "Which university did the alleged author of The Glass Harbor attend?",
        False,
        None,
        "uncertainty",
    ),
    HeldoutCase(
        "H19",
        "Which university did the author of The Blue Lantern attend, if any?",
        False,
        None,
        "qualification",
    ),
    HeldoutCase(
        "H20",
        "Which university did the author of The Winter Orchard attend before teaching there?",
        False,
        None,
        "compound_relation",
    ),

    # Unsupported or additional information requests.
    HeldoutCase(
        "H21",
        "Which university did the author of The Amber Compass attend, and in what year?",
        False,
        None,
        "compound_question",
    ),
    HeldoutCase(
        "H22",
        "Which university did the author of The Glass Harbor attend, and who was their advisor?",
        False,
        None,
        "compound_question",
    ),
    HeldoutCase(
        "H23",
        "Which city contains the university attended by the author of The Blue Lantern?",
        False,
        None,
        "unsupported_predicate",
    ),
    HeldoutCase(
        "H24",
        "Who wrote The Winter Orchard?",
        False,
        None,
        "out_of_scope_one_hop",
    ),
)


def check_semantic_fidelity(requirements, expected_title):
    """Check the intended two-hop graph, allowing variable renaming."""

    relations = requirements.relationships

    if len(relations) != 2:
        return False, "wrong_relationship_count"

    authored = [
        r for r in relations
        if r.predicate == "authored_by"
    ]
    attended = [
        r for r in relations
        if r.predicate == "attended"
    ]

    if len(authored) != 1 or len(attended) != 1:
        return False, "wrong_predicates"

    first = authored[0]
    second = attended[0]

    def normalize(value):
        return " ".join(value.split()).strip().casefold()

    if normalize(first.subject) != normalize(expected_title):
        return False, "wrong_book"

    if not first.object.startswith("?"):
        return False, "missing_author_variable"

    if second.subject != first.object:
        return False, "broken_join"

    if not second.object.startswith("?"):
        return False, "missing_education_variable"

    if second.object == first.object:
        return False, "same_author_answer_variable"

    if requirements.answer_variable != second.object:
        return False, "wrong_answer_target"

    return True, "correct"


def classify(case, result):
    accepted = (
        result.status == QwenDecompositionStatus.VALID_PROPOSAL
    )

    if not accepted:
        return (
            "missed_supported"
            if case.expected_supported
            else "correct_rejection"
        )

    if not case.expected_supported:
        return "false_acceptance"

    if result.validation is None:
        return "incorrect_decomposition:missing_validation"

    requirements = result.validation.requirements

    if requirements is None:
        return "incorrect_decomposition:missing_requirements"

    correct, reason = check_semantic_fidelity(
        requirements,
        case.expected_title,
    )

    if correct:
        return "correct_decomposition"

    return f"incorrect_decomposition:{reason}"


def main():
    print("Phase 11.4G-9: held-out adversarial evaluation")
    print("Loading Qwen once...", flush=True)

    decomposer = QwenQuestionDecomposer()

    records = []

    for index, case in enumerate(CASES, start=1):
        print("\n" + "=" * 68, flush=True)
        print(
            f"[{index}/{len(CASES)}] {case.case_id} "
            f"[{case.category}]",
            flush=True,
        )
        print(f"Question: {case.question}", flush=True)

        result = decomposer.decompose(case.question)
        outcome = classify(case, result)

        print(f"Expected supported: {case.expected_supported}")
        print(f"Status: {result.status.value}")
        print(f"Outcome: {outcome}")
        print(f"Explanation: {result.explanation}")
        print(f"Raw response:\n{result.raw_response}")

        records.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "expected_supported": case.expected_supported,
                "expected_title": case.expected_title,
                "status": result.status.value,
                "outcome": outcome,
                "explanation": result.explanation,
                "raw_response": result.raw_response,
                "proposal": result.proposal,
            }
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "experiment": "phase11_4g9",
                    "completed_cases": len(records),
                    "total_cases": len(CASES),
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    supported = sum(c.expected_supported for c in CASES)
    unsupported = len(CASES) - supported

    correct_supported = sum(
        r["outcome"] == "correct_decomposition"
        for r in records
    )
    correct_rejections = sum(
        r["outcome"] == "correct_rejection"
        for r in records
    )
    false_acceptances = sum(
        r["outcome"] == "false_acceptance"
        for r in records
    )
    missed_supported = sum(
        r["outcome"] == "missed_supported"
        for r in records
    )
    incorrect = sum(
        r["outcome"].startswith("incorrect_decomposition:")
        for r in records
    )

    print("\n" + "=" * 68)
    print("PHASE 11.4G-9 — HELD-OUT RESULTS")
    print(f"Total questions: {len(CASES)}")
    print(
        f"Correct supported: {correct_supported}/{supported}"
    )
    print(
        f"Correct rejections: {correct_rejections}/{unsupported}"
    )
    print(f"Missed supported: {missed_supported}")
    print(f"Incorrect accepted decompositions: {incorrect}")
    print(f"False acceptances: {false_acceptances}")
    print(
        "Overall correct decisions: "
        f"{correct_supported + correct_rejections}/{len(CASES)}"
    )

    print("\nPER-CATEGORY RESULTS")
    for category in sorted({c.category for c in CASES}):
        subset = [
            r for r in records
            if r["category"] == category
        ]
        correct = sum(
            r["outcome"] in (
                "correct_decomposition",
                "correct_rejection",
            )
            for r in subset
        )
        print(f"  {category}: {correct}/{len(subset)}")

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
