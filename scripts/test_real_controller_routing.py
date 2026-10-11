"""Phase 10.5D: frozen Phase 9 detector + real retrieval + safe controller routing.

Run: python -m scripts.test_real_controller_routing --max-questions 5
Recovery executor is deliberately a deterministic probe, NOT Qwen recovery.
Oracle labels are used only for post-decision offline metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_scaled_diagnostic_dataset import (
    CROSS_ENCODER_MODEL_NAME, DENSE_MODEL_NAME, EMBEDDINGS_PATH,
    EXPECTED_CHUNKS, EXPECTED_DIMENSION, SPLIT_CONFIGS,
    load_frozen_chunks, load_query_examples,
)
from src.diagnosis.runtime import RUNTIME_FEATURES, RuntimeSufficiencyDiagnoser
from src.embeddings.dense_embedder import DenseEmbedder, DenseEmbeddingConfig
from src.recovery.controller import ControllerDecision, ControllerOutcome, RecoveryController
from src.recovery.runtime_integration import DiagnosedRecoveryController
from src.recovery.states import RecoveryResult, RecoveryStatus
from src.retrieval.dense_retriever import DenseRetriever, format_chunk_for_retrieval
from src.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from src.retrieval.sparse_retriever import BM25Retriever

TRAIN_CSV = Path('artifacts/diagnosis/scaled/diagnostic_train.csv')
VALIDATION_CSV = Path('artifacts/diagnosis/scaled/diagnostic_validation.csv')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-questions', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_questions <= 200:
        parser.error('--max-questions must be between 1 and 200')

    train = pd.read_csv(TRAIN_CSV)
    validation = pd.read_csv(VALIDATION_CSV)
    required = set(RUNTIME_FEATURES) | {'question_index', 'question', 'split', 'is_evidence_sufficient'}
    for name, frame, expected_len, split in (
        ('train', train, 1000, 'train'),
        ('validation', validation, 200, 'validation'),
    ):
        missing = required - set(frame.columns)
        if missing or len(frame) != expected_len or set(frame['split'].astype(str)) != {split}:
            raise RuntimeError(f'Invalid frozen {name} dataset: missing={sorted(missing)}')
        if frame['question_index'].duplicated().any():
            raise RuntimeError(f'Duplicate {name} question indices')

    print('Fitting unchanged Phase 9 detector on training split only...', flush=True)
    diagnoser = RuntimeSufficiencyDiagnoser()
    diagnoser.fit(train.loc[:, RUNTIME_FEATURES], train['is_evidence_sufficient'].astype(int))
    if abs(diagnoser.threshold - 0.58) > 1e-12:
        raise RuntimeError('Frozen threshold changed')

    print('Loading frozen retrieval pipeline...', flush=True)
    chunks = load_frozen_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != EXPECTED_CHUNKS or embeddings.shape != (EXPECTED_CHUNKS, EXPECTED_DIMENSION) or embeddings.dtype != np.float32:
        raise RuntimeError('Frozen corpus/embedding mismatch')
    dense = DenseRetriever(chunks=chunks, embeddings=embeddings)
    sparse = BM25Retriever(chunks=chunks, retrieval_texts=[format_chunk_for_retrieval(c) for c in chunks])
    embedder = DenseEmbedder(DenseEmbeddingConfig(model_name=DENSE_MODEL_NAME, batch_size=32, normalize_embeddings=True))
    reranker = CrossEncoderReranker(
        dense_retriever=dense, sparse_retriever=sparse,
        config=RerankerConfig(model_name=CROSS_ENCODER_MODEL_NAME, candidate_k=20, batch_size=32),
    )

    requests = []
    def probe_executor(request):
        requests.append(request)
        # Simulate a completed recovery with usable evidence. This is NOT a real recovery run.
        return RecoveryResult(
            action=request.action, status=RecoveryStatus.COMPLETED,
            original_query=request.original_query, recovery_queries=('probe-only',),
            original_results=request.original_results,
            recovered_results=request.original_results[:1],
        )

    integrated = DiagnosedRecoveryController(diagnoser, RecoveryController(probe_executor))
    examples = load_query_examples('validation')
    if len(examples) != 200:
        raise RuntimeError('Expected 200 validation examples')
    indexed = validation.set_index('question_index')
    start = SPLIT_CONFIGS['validation']['start']
    routed = 0
    correct = 0
    false_flags = 0
    missed_insufficient = 0
    print('\nPHASE 10.5D — REAL DETECTOR AND CONTROLLER ROUTING', flush=True)
    for position, example in enumerate(examples[:args.max_questions], 1):
        index = start + position - 1
        saved = indexed.loc[index]
        if str(saved['question']) != str(example.question):
            raise RuntimeError(f'Question alignment mismatch at {index}')
        results = tuple(reranker.search(query=example.question, embedder=embedder, top_k=10))
        if len(results) != 10:
            raise RuntimeError(f'Unexpected first-hop count at {index}')
        before = len(requests)
        outcome = integrated.run(query=example.question, original_results=results)
        decision = outcome.controller_result
        predicted = outcome.diagnosis.predicted_sufficient
        expected_recovery = not predicted
        if outcome.diagnosis.diagnosis.recovery_recommended != expected_recovery:
            raise AssertionError(f'Diagnosis and prediction disagree at {index}')
        if decision.final_results != results or decision.original_results != results:
            raise AssertionError(f'Original evidence not preserved at {index}')
        if expected_recovery:
            routed += 1
            if len(requests) != before + 1 or decision.decision != ControllerDecision.ATTEMPT_RECOVERY or decision.outcome != ControllerOutcome.RECOVERY_PENDING_VERIFICATION:
                raise AssertionError(f'Recovery routing incorrect at {index}')
        else:
            if len(requests) != before or decision.decision != ControllerDecision.KEEP_ORIGINAL or decision.outcome != ControllerOutcome.ORIGINAL_ACCEPTED:
                raise AssertionError(f'Keep routing incorrect at {index}')
        # Offline evaluation only: label never enters runtime features, detector, or controller.
        actual_sufficient = bool(int(saved['is_evidence_sufficient']))
        correct += predicted == actual_sufficient
        false_flags += expected_recovery and actual_sufficient
        missed_insufficient += predicted and not actual_sufficient
        print(f'[{position}/{args.max_questions}] index={index} P(sufficient)={outcome.diagnosis.sufficient_probability:.4f} route={decision.decision.value} outcome={decision.outcome.value}', flush=True)

    n = args.max_questions
    print(f'\nQuestions: {n}; routed to probe recovery: {routed}; kept original: {n-routed}')
    print(f'Offline sufficiency classification: {correct}/{n} correct')
    print(f'Offline false recovery flags: {false_flags}; missed insufficient: {missed_insufficient}')
    print('PASS: real retrieval + frozen detector + safe controller routing.')
    print('NOTE: probe recovery does not measure recovery quality or verification success.')
    print('PHASE 10.5D ROUTING TEST PASSED.')


if __name__ == '__main__':
    main()
