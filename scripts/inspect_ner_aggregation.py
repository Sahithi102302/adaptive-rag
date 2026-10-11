from __future__ import annotations

from transformers import pipeline


MODEL_NAME = "dslim/bert-base-NER"


TEXTS = [
    (
        "HECTOR",
        "Héctor Canziani was an Argentine poet, screenwriter and "
        "film director. He is best known for his directorship and "
        "production of the 1950 tango dancing film Al Compás de tu "
        "Mentira based on a play by Oscar Wilde.",
    ),
    (
        "MING_NA",
        "The series also starred Joey Slotnick as Eliot's best "
        "friend Sam Sloan, Ming-Na Wen as Sam's wife Trudy and "
        "Ernest Borgnine as doorman Manny.",
    ),
    (
        "YEMI_ALADE",
        "King of Queens is the debut studio album by Nigerian "
        "singer Yemi Alade, released by Effyzzie Music Group.",
    ),
    (
        "RATCHANOK",
        "Wang Yihan was the defending champion. "
        "Ratchanok Inthanon defeated Li Xuerui in the final.",
    ),
]


def print_predictions(
    name: str,
    ner,
    text: str,
) -> None:
    print(f"\n{name}")

    predictions = ner(text)

    for prediction in predictions:
        word = prediction.get(
            "word",
            prediction.get(
                "text",
                "",
            ),
        )

        label = prediction.get(
            "entity_group",
            prediction.get(
                "entity",
                "UNKNOWN",
            ),
        )

        score = float(
            prediction.get(
                "score",
                0.0,
            )
        )

        print(
            f"  entity={word!r} "
            f"label={label!r} "
            f"score={score:.4f}"
        )


def main() -> None:
    print("=" * 80)
    print(
        "PHASE 10.3G-C — NER AGGREGATION ABLATION"
    )
    print("=" * 80)

    print(
        "\nLoading SIMPLE aggregation..."
    )

    simple_ner = pipeline(
        "ner",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )

    print(
        "Loading FIRST aggregation..."
    )

    first_ner = pipeline(
        "ner",
        model=MODEL_NAME,
        aggregation_strategy="first",
    )

    for case_name, text in TEXTS:
        print(
            "\n" + "=" * 80
        )

        print(
            f"CASE: {case_name}"
        )

        print(
            "=" * 80
        )

        print_predictions(
            "SIMPLE",
            simple_ner,
            text,
        )

        print_predictions(
            "FIRST",
            first_ner,
            text,
        )

    print(
        "\nPHASE 10.3G-C INSPECTION COMPLETE."
    )


if __name__ == "__main__":
    main()