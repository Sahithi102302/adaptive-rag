"""Phase 10.5C: compare real runtime features with frozen Phase 9 validation rows.

Run: python -m scripts.test_real_feature_parity --max-questions 5
No model fitting, Qwen, gold labels, or oracle routing.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_scaled_diagnostic_dataset import (
    CROSS_ENCODER_MODEL_NAME,
    DENSE_MODEL_NAME,
    EMBEDDINGS_PATH,
    EXPECTED_CHUNKS,
    EXPECTED_DIMENSION,
    SPLIT_CONFIGS,
    load_frozen_chunks,
    load_query_examples,
)
from src.diagnosis.runtime import RUNTIME_FEATURES
from src.embeddings.dense_embedder import DenseEmbedder, DenseEmbeddingConfig
from src.recovery.runtime_integration import build_runtime_feature_row
from src.retrieval.dense_retriever import DenseRetriever, format_chunk_for_retrieval
from src.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from src.retrieval.sparse_retriever import BM25Retriever

VALIDATION_CSV = Path('artifacts/diagnosis/scaled/diagnostic_validation.csv')
CANDIDATE_K = 20
FINAL_K = 10
BATCH_SIZE = 32


def same_number(actual: object, expected: object, *, atol: float) -> bool:
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    try:
        a = float(actual)
        b = float(expected)
    except (TypeError, ValueError):
        return actual == expected
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-questions', type=int, default=5)
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    if not 1 <= args.max_questions <= 200:
        parser.error('--max-questions must be 1..200')
    if args.atol < 0:
        parser.error('--atol must be nonnegative')
    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(f'Missing frozen validation CSV: {VALIDATION_CSV}')

    frame = pd.read_csv(VALIDATION_CSV)
    required = {'question_index', 'question', 'split', *RUNTIME_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f'Validation CSV missing columns: {sorted(missing)}')
    if len(frame) != 200 or set(frame['split'].astype(str)) != {'validation'}:
        raise RuntimeError('Expected 200 frozen validation rows, split=validation')
    if frame['question_index'].duplicated().any():
        raise RuntimeError('Duplicate validation question indices')
    by_index = frame.set_index('question_index', verify_integrity=True)
    examples = load_query_examples('validation')
    if len(examples) != 200:
        raise RuntimeError(f'Expected 200 validation examples, got {len(examples)}')

    print('Loading frozen corpus, embeddings, and retrieval models...', flush=True)
    chunks = load_frozen_chunks()
    if len(chunks) != EXPECTED_CHUNKS:
        raise RuntimeError('Frozen corpus size mismatch')
    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.shape != (EXPECTED_CHUNKS, EXPECTED_DIMENSION) or embeddings.dtype != np.float32:
        raise RuntimeError('Frozen embeddings mismatch')
    dense = DenseRetriever(chunks=chunks, embeddings=embeddings)
    sparse = BM25Retriever(
        chunks=chunks,
        retrieval_texts=[format_chunk_for_retrieval(c) for c in chunks],
    )
    embedder = DenseEmbedder(DenseEmbeddingConfig(
        model_name=DENSE_MODEL_NAME,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    ))
    reranker = CrossEncoderReranker(
        dense_retriever=dense,
        sparse_retriever=sparse,
        config=RerankerConfig(
            model_name=CROSS_ENCODER_MODEL_NAME,
            candidate_k=CANDIDATE_K,
            batch_size=BATCH_SIZE,
        ),
    )

    start_index = SPLIT_CONFIGS['validation']['start']
    failures: list[tuple[int, str, object, object]] = []
    print('\nPHASE 10.5C — REAL FEATURE PARITY', flush=True)
    for position, example in enumerate(examples[:args.max_questions], start=1):
        absolute_index = start_index + position - 1
        if absolute_index not in by_index.index:
            raise RuntimeError(f'Missing frozen question_index={absolute_index}')
        saved = by_index.loc[absolute_index]
        if str(saved['question']) != str(example.question):
            raise RuntimeError(f'Question text mismatch at index {absolute_index}')
        results = reranker.search(query=example.question, embedder=embedder, top_k=FINAL_K)
        if len(results) != FINAL_K:
            raise RuntimeError(f'Expected {FINAL_K} results for index {absolute_index}')
        features = build_runtime_feature_row(query=example.question, results=results)
        if list(features) != list(RUNTIME_FEATURES):
            raise RuntimeError('Feature order mismatch')
        row_failures = []
        for name in RUNTIME_FEATURES:
            if not same_number(features[name], saved[name], atol=args.atol):
                row_failures.append((absolute_index, name, features[name], saved[name]))
        failures.extend(row_failures)
        print(f'[{position}/{args.max_questions}] index={absolute_index} '
              f'feature_matches={len(RUNTIME_FEATURES)-len(row_failures)}/{len(RUNTIME_FEATURES)}',
              flush=True)

    print(f'\nCompared {args.max_questions * len(RUNTIME_FEATURES)} feature values.')
    if failures:
        print(f'FAIL: {len(failures)} mismatches; first 20:')
        for index, name, actual, expected in failures[:20]:
            print(f'  index={index} {name}: runtime={actual!r} frozen={expected!r}')
        raise SystemExit(1)
    print('PASS: all real runtime features match frozen validation rows.')
    print('PHASE 10.5C REAL FEATURE PARITY PASSED.')


if __name__ == '__main__':
    main()
