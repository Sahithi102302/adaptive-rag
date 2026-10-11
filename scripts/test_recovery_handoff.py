from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from src.recovery.controller import RecoveryController, ControllerOutcome
from src.recovery.evidence_selection import EvidencePreservingSelector
from src.recovery.recovery_handoff import RecoveryHandoffBuilder
from src.recovery.states import RecoveryResult, RecoveryStatus


def make_result(chunk_id, rank):
    return SimpleNamespace(rank=rank, chunk=SimpleNamespace(id=chunk_id, text=chunk_id, document_id=chunk_id))


def main():
    original = tuple(make_result(f"old-{i}", i + 1) for i in range(10))
    recovered = tuple(make_result(f"new-{i}", i + 1) for i in range(10))
    def execute(request):
        return RecoveryResult(
            action=request.action, status=RecoveryStatus.COMPLETED,
            original_query=request.original_query, recovery_queries=("bridge query",),
            original_results=request.original_results, recovered_results=recovered,
        )
    diagnosis = lambda recover: SimpleNamespace(diagnosis=SimpleNamespace(recovery_recommended=recover))
    controller = RecoveryController(execute)
    builder = RecoveryHandoffBuilder(EvidencePreservingSelector())

    kept = controller.run("question", original, diagnosis(False))
    kept_handoff = builder.build(kept)
    assert not kept_handoff.pending_verification and kept_handoff.selection is None
    assert kept_handoff.original_results == original
    print("PASS: no recovery produces no verification candidate")

    pending = controller.run("question", original, diagnosis(True))
    assert pending.outcome == ControllerOutcome.RECOVERY_PENDING_VERIFICATION
    handoff = builder.build(pending)
    assert handoff.pending_verification
    assert handoff.original_results == original
    assert pending.final_results == original
    selected = handoff.selection.selected_results
    assert len(selected) == 10
    assert len({item.chunk_id for item in selected}) == 10
    assert sum(item.first_hop_rank is not None for item in selected) == 5
    assert sum(item.second_hop_rank is not None for item in selected) == 5
    print("PASS: 5/5 candidate selection retains provenance and original fallback")

    invalid = replace(pending, final_results=recovered)
    try:
        builder.build(invalid)
    except ValueError:
        print("PASS: premature evidence replacement rejected")
    else:
        raise AssertionError("Premature replacement was not rejected")

    bad_recovery = replace(pending.recovery_result, original_results=recovered)
    try:
        builder.build(replace(pending, recovery_result=bad_recovery))
    except ValueError:
        print("PASS: mismatched recovery provenance rejected")
    else:
        raise AssertionError("Mismatched provenance was not rejected")

    print("PHASE 10.6 RECOVERY HANDOFF TESTS PASSED")


if __name__ == "__main__":
    main()
