from __future__ import annotations

from dataclasses import make_dataclass
from unittest.mock import Mock, patch

from src.diagnosis.runtime import RUNTIME_FEATURES
from src.recovery.runtime_integration import (
    DiagnosedRecoveryController,
    build_runtime_feature_row,
)


def main() -> None:
    retrieval_names = RUNTIME_FEATURES[:19]
    lexical_names = RUNTIME_FEATURES[19:]
    assert len(RUNTIME_FEATURES) == 29
    assert len(retrieval_names) == 19 and len(lexical_names) == 10

    Retrieval = make_dataclass("Retrieval", [(n, object) for n in retrieval_names])
    Lexical = make_dataclass("Lexical", [(n, object) for n in lexical_names])
    results = (object(), object())
    with (
        patch(
            "src.recovery.runtime_integration.extract_runtime_signals",
            return_value=Retrieval(*range(19)),
        ) as retrieval_fn,
        patch(
            "src.recovery.runtime_integration.extract_lexical_evidence_signals",
            return_value=Lexical(*range(19, 29)),
        ) as lexical_fn,
    ):
        row = build_runtime_feature_row("question", results)
        assert list(row) == RUNTIME_FEATURES
        assert list(row.values()) == list(range(29))
        retrieval_fn.assert_called_once_with(results)
        lexical_fn.assert_called_once_with(query="question", results=results)
        print("PASS: frozen feature ordering and extraction interfaces")

        diagnosis = object()
        controller_result = object()
        diagnoser = Mock()
        diagnoser.diagnose.return_value = diagnosis
        controller = Mock()
        controller.run.return_value = controller_result
        pipeline = DiagnosedRecoveryController(diagnoser, controller)
        output = pipeline.run("question", results)
        diagnoser.diagnose.assert_called_once_with(row)
        controller.run.assert_called_once_with(
            query="question", original_results=results, diagnosis=diagnosis
        )
        assert output.feature_row == row
        assert output.diagnosis is diagnosis
        assert output.controller_result is controller_result
        print("PASS: runtime detector feeds controller without gold or oracle")

    print("PHASE 10.5B INTEGRATION UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
