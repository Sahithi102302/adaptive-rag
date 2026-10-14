
"""Phase 11.4G-12H-10C: question structure tests."""

from src.recovery.question_structure_diagnostic import (
    QuestionStructure,
    QuestionStructureDiagnostic,
)


CASES = [
    (
        "S01",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        QuestionStructure.EXPLICIT_COMPARISON,
        ("comparison",),
    ),
    (
        "S02",
        "Who is older, Annie Morton or Terry Richardson?",
        QuestionStructure.EXPLICIT_COMPARISON,
        ("comparison",),
    ),
    (
        "S03",
        "What government position was held by the woman who portrayed Corliss Archer?",
        QuestionStructure.POSSIBLE_BRIDGE,
        (),
    ),
    (
        "S04",
        "The director of Big Stone Gap is based in what city?",
        QuestionStructure.POSSIBLE_BRIDGE,
        (),
    ),
    (
        "S05",
        "Which university did the author of The Copper Horizon attend?",
        QuestionStructure.POSSIBLE_BRIDGE,
        (),
    ),
    (
        "S06",
        "Which university did the author of The Copper Horizon never attend?",
        QuestionStructure.POSSIBLE_BRIDGE,
        ("negation",),
    ),
    (
        "S07",
        "What happened in 1998?",
        QuestionStructure.UNRESOLVED,
        ("temporal",),
    ),
    (
        "S08",
        "",
        QuestionStructure.UNRESOLVED,
        (),
    ),
    (
    "S09",
    (
        "Is the Pakistan fast bowler who joined the Kent County "
        "Cricket Club in June, 2011 a left-hand or right-hand batsmans?"
    ),
    QuestionStructure.POSSIBLE_BRIDGE,
    ("temporal",),
    ),
    (
    "S10",
    "Who died first, George Archainbaud or Ralph Murphy?",
    QuestionStructure.EXPLICIT_COMPARISON,
    ("superlative",),
),
(
    "S11",
    "Which tower is taller, 712 Fifth Avenue or Manhattan Life Insurance Building?",
    QuestionStructure.EXPLICIT_COMPARISON,
    (),
),
(
    "S12",
    "Which film was Oscar nominated, LaLee's Kin: The Legacy of Cotton or Gimme Shelter?",
    QuestionStructure.EXPLICIT_COMPARISON,
    (),
),
(
    "S13",
    "Who has written more than 300 papers, Semyon Aranovich Gershgorin or Pavel Alexandrov?",
    QuestionStructure.EXPLICIT_COMPARISON,
    (),
),
]


def main() -> None:
    diagnostic = QuestionStructureDiagnostic()
    passed = 0

    for case_id, question, expected_structure, expected_constraints in CASES:
        result = diagnostic.analyze(question)

        ok = (
            result.structure == expected_structure
            and result.constraints == expected_constraints
        )

        passed += int(ok)

        print(
            f"{'PASS' if ok else 'FAIL'} {case_id}: "
            f"structure={result.structure.value}, "
            f"constraints={result.constraints}, "
            f"reason={result.reason}"
        )

    print(f"\nQuestion structure diagnostics: {passed}/{len(CASES)}")

    if passed != len(CASES):
        raise AssertionError("Question structure diagnostic tests failed.")


if __name__ == "__main__":
    main()
