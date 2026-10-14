
"""Phase 11.4G-11: frozen-pipeline adversarial evaluation.

Tests Qwen + structural validation + Semantic Gate V2.

30 new hand-authored cases:
- 15 supported
- 15 unsupported under the restricted two-hop task

The existing model, prompt, and gate must remain unchanged.

This is a diagnostic adversarial evaluation, not a statistically
representative benchmark or proof of general semantic fidelity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.recovery.qwen_question_decomposer import (
    QwenDecompositionStatus,
    QwenQuestionDecomposer,
)


OUTPUT_PATH = Path(
    "outputs/phase11_4g11_adversarial_v2_results.json"
)


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    question: str
    expected_supported: bool
    expected_title: str | None
    category: str


CASES = (
    # -----------------------------------------
    # Supported: new titles and familiar forms
    # -----------------------------------------
    AdversarialCase(
        "A01",
        "Which university did the author of The Crimson Observatory attend?",
        True,
        "The Crimson Observatory",
        "new_entity",
    ),
    AdversarialCase(
        "A02",
        "Where did the writer of A Map of Forgotten Rivers study?",
        True,
        "A Map of Forgotten Rivers",
        "new_entity",
    ),
    AdversarialCase(
        "A03",
        "Which college was attended by the author of The Copper Horizon?",
        True,
        "The Copper Horizon",
        "passive_voice",
    ),
    AdversarialCase(
        "A04",
        "What university did the person who wrote The Last Meridian go to?",
        True,
        "The Last Meridian",
        "relative_clause",
    ),
    AdversarialCase(
        "A05",
        "Name the college attended by the writer of The Paper Constellation.",
        True,
        "The Paper Constellation",
        "imperative",
    ),

    # -----------------------------------------
    # Supported: unseen meaning-preserving
    # paraphrases
    # -----------------------------------------
    AdversarialCase(
        "A06",
        "Which university did the novelist behind The Copper Horizon attend?",
        True,
        "The Copper Horizon",
        "unseen_paraphrase",
    ),
    AdversarialCase(
        "A07",
        "At what college was the author of The Last Meridian educated?",
        True,
        "The Last Meridian",
        "unseen_paraphrase",
    ),
    AdversarialCase(
        "A08",
        "Where did the person who authored The Paper Constellation go to college?",
        True,
        "The Paper Constellation",
        "unseen_paraphrase",
    ),
    AdversarialCase(
        "A09",
        "Identify the university where the writer of The Crimson Observatory studied.",
        True,
        "The Crimson Observatory",
        "unseen_paraphrase",
    ),
    AdversarialCase(
        "A10",
        "Which college did The Copper Horizon's author attend?",
        True,
        "The Copper Horizon",
        "unseen_paraphrase",
    ),

    # -----------------------------------------
    # Supported: unusual but valid book titles
    # -----------------------------------------
    AdversarialCase(
        "A11",
        "Which university did the author of How to Attend University attend?",
        True,
        "How to Attend University",
        "entity_boundary",
    ),
    AdversarialCase(
        "A12",
        "Where did the author of The Author of Tomorrow study?",
        True,
        "The Author of Tomorrow",
        "entity_boundary",
    ),
    AdversarialCase(
        "A13",
        'Which university did the writer of "Which University?" attend?',
        True,
        "Which University?",
        "entity_boundary",
    ),
    AdversarialCase(
        "A14",
        "Which university did the author of The University of Dreams attend?",
        True,
        "The University of Dreams",
        "entity_boundary",
    ),
    AdversarialCase(
        "A15",
        "The writer of The College Years studied at which university?",
        True,
        "The College Years",
        "entity_boundary",
    ),

    # -----------------------------------------
    # Unsupported: semantic substitutions
    # -----------------------------------------
    AdversarialCase(
        "A16",
        "Which university employed the author of The Copper Horizon?",
        False,
        None,
        "relation_substitution",
    ),
    AdversarialCase(
        "A17",
        "Which university awarded a degree to the writer of The Last Meridian?",
        False,
        None,
        "relation_substitution",
    ),
    AdversarialCase(
        "A18",
        "Which university did the narrator of The Paper Constellation attend?",
        False,
        None,
        "role_substitution",
    ),
    AdversarialCase(
        "A19",
        "Which university did the co-author of The Crimson Observatory attend?",
        False,
        None,
        "role_substitution",
    ),
    AdversarialCase(
        "A20",
        "Which university did the fictional author in The Copper Horizon attend?",
        False,
        None,
        "role_substitution",
    ),

    # -----------------------------------------
    # Unsupported: negation and uncertainty
    # -----------------------------------------
    AdversarialCase(
        "A21",
        "Which university did the author of The Last Meridian never attend?",
        False,
        None,
        "negation",
    ),
    AdversarialCase(
        "A22",
        "Which university might the author of The Paper Constellation have attended?",
        False,
        None,
        "uncertainty",
    ),
    AdversarialCase(
        "A23",
        "Which university did the supposed author of The Copper Horizon attend?",
        False,
        None,
        "uncertainty",
    ),

    # -----------------------------------------
    # Unsupported: additional constraints
    # -----------------------------------------
    AdversarialCase(
        "A24",
        "Which university did the author of The Crimson Observatory attend first?",
        False,
        None,
        "additional_constraint",
    ),
    AdversarialCase(
        "A25",
        "Which university did the author of The Last Meridian attend in 1998?",
        False,
        None,
        "additional_constraint",
    ),
    AdversarialCase(
        "A26",
        "Which university did the author of The Paper Constellation attend before moving abroad?",
        False,
        None,
        "additional_constraint",
    ),
    AdversarialCase(
        "A27",
        "Which university did the author of The Copper Horizon attend with their sibling?",
        False,
        None,
        "additional_constraint",
    ),

    # -----------------------------------------
    # Unsupported: composition and scope
    # -----------------------------------------
    AdversarialCase(
        "A28",
        "Which university did the authors of The Crimson Observatory attend?",
        False,
        None,
        "multiple_authors",
    ),
    AdversarialCase(
        "A29",
        "Which university did the author of The Last Meridian attend, and what degree did they earn?",
        False,
        None,
        "compound_question",
    ),
    AdversarialCase(
        "A30",
        "Who authored The Paper Constellation?",
        False,
        None,
        "out_of_scope_one_hop",
    ),
)


def normalize(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def semantic_fidelity(requirements, expected_title):
    """Check a two-hop graph without assuming variable names."""

    relations = requirements.relationships

    if len(relations) != 2:
        return False, "wrong_relationship_count"

    authored = [
        relation
        for relation in relations
        if relation.predicate == "authored_by"
    ]
    attended = [
        relation
        for relation in relations
        if relation.predicate == "attended"
    ]

    if len(authored) != 1 or len(attended) != 1:
        return False, "wrong_predicates"

    first = authored[0]
    second = attended[0]

    if normalize(first.subject) != normalize(expected_title):
        return False, "wrong_book_entity"

    if not first.object.startswith("?"):
        return False, "missing_author_variable"

    if second.subject != first.object:
        return False, "broken_join"

    if not second.object.startswith("?"):
        return False, "missing_answer_variable"

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

    correct, reason = semantic_fidelity(
        requirements,
        case.expected_title,
    )

    return (
        "correct_decomposition"
        if correct
        else f"incorrect_decomposition:{reason}"
    )


def main():
    assert len(CASES) == 30
    assert len({case.case_id for case in CASES}) == 30
    assert sum(case.expected_supported for case in CASES) == 15

    print("PHASE 11.4G-11 — ADVERSARIAL EVALUATION")
    print("Frozen Qwen + Semantic Gate V2")
    print("Loading Qwen once...", flush=True)

    decomposer = QwenQuestionDecomposer()

    records = []

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, case in enumerate(CASES, start=1):
        print("\n" + "=" * 68)
        print(f"[{index}/30] {case.case_id}: {case.question}")
        print(f"Category: {case.category}")
        print(f"Expected supported: {case.expected_supported}")

        result = decomposer.decompose(case.question)
        outcome = classify(case, result)

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

        # Save after each question.
        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "experiment": "phase11_4g11",
                    "pipeline": "qwen_semantic_gate_v2",
                    "completed_cases": len(records),
                    "total_cases": len(CASES),
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    supported = sum(
        case.expected_supported
        for case in CASES
    )
    unsupported = len(CASES) - supported

    correct_supported = sum(
        record["outcome"] == "correct_decomposition"
        for record in records
    )

    correct_rejections = sum(
        record["outcome"] == "correct_rejection"
        for record in records
    )

    missed_supported = sum(
        record["outcome"] == "missed_supported"
        for record in records
    )

    false_acceptances = sum(
        record["outcome"] == "false_acceptance"
        for record in records
    )

    incorrect_decompositions = sum(
        record["outcome"].startswith(
            "incorrect_decomposition:"
        )
        for record in records
    )

    print("\n" + "=" * 68)
    print("PHASE 11.4G-11 — FINAL RESULTS")
    print(f"Total questions: {len(CASES)}")
    print(f"Supported questions: {supported}")
    print(f"Unsupported questions: {unsupported}")
    print(f"Correct decompositions: {correct_supported}/{supported}")
    print(f"Correct rejections: {correct_rejections}/{unsupported}")
    print(f"Missed supported: {missed_supported}")
    print(f"False acceptances: {false_acceptances}")
    print(f"Incorrect decompositions: {incorrect_decompositions}")
    print(
        "Overall correct decisions: "
        f"{correct_supported + correct_rejections}/{len(CASES)}"
    )

    print("\nPER-CATEGORY RESULTS")

    for category in sorted({case.category for case in CASES}):
        subset = [
            record
            for record in records
            if record["category"] == category
        ]

        correct = sum(
            record["outcome"] in (
                "correct_decomposition",
                "correct_rejection",
            )
            for record in subset
        )

        print(f"  {category}: {correct}/{len(subset)}")

    print(f"\nSaved results: {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
