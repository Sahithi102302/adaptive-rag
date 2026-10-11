# Phase 10.4 — Evidence-Preserving Final Context Selection

**Status:** Diagnostic experiments completed (10.4A–10.4C); policy selection provisional.  
**Dataset:** HotpotQA validation; first 10 oracle-selected evidence-coverage failures from the frozen retrieval pipeline.  
**Scope:** Offline evaluation. Gold facts were used for case selection and scoring, not for runtime candidate selection or query generation.

## Research question

Can a bounded final top-10 context combine first-hop and second-hop retrieval evidence without dropping already-retrieved supporting facts?

## Frozen upstream pipeline

- First-hop: frozen dense + BM25 candidate retrieval, cross-encoder reranking, candidate_k=20, final_k=10.
- Runtime second hop: content NER, closed-candidate Qwen bridge selection, constrained relation-query generation, and independent second-hop retrieval at top_k=10.
- Expected-bridge condition: manually specified bridge candidates used **only** for an offline diagnostic intervention, not a deployable runtime result.
- Original and second-hop retrieval outputs remained separate until final selection.

## 10.4A — Selection interface

Implemented `src/recovery/evidence_selection.py` with chunk-ID deduplication, provenance (first hop / second hop / both), deterministic selection, and configurable first/second-hop reservation preferences. Synthetic unit tests passed. The reservation numbers are preferences, not strict unique-source quotas when hops overlap.

## 10.4B — Synthetic comparison

Compared first-hop-only, naive first-hop-priority union, and hop-balanced selection. All three controlled scenarios behaved as designed:

1. Successful repair: hop balancing included a missing fact without losing original facts.
2. Recovery harm: hop balancing gained one fact but displaced another necessary original fact; completeness remained false.
3. Overlapping retrieval: deduplication allowed useful second-hop evidence into the bounded context.

These scenarios test implementation behavior; they do not estimate real-world performance.

## 10.4C — Real diagnostic results

### Condition A — Actual runtime (10 selected failures, 23 total gold supporting facts)

| Policy | Gold facts retrieved | Recall | Complete cases | Newly recovered facts | Original facts lost | Repaired cases |
|---|---:|---:|---:|---:|---:|---:|
| First-hop only | 12/23 | 0.5217 | 0/10 | 0 | 0 | 0 |
| Naive rank union | 12/23 | 0.5217 | 0/10 | 0 | 0 | 0 |
| Hop-balanced 5/5 | 14/23 | 0.6087 | 2/10 | 2 | 0 | 2 |
| Hop-balanced 7/3 | 14/23 | 0.6087 | 2/10 | 2 | 0 | 2 |
| Hop-balanced 3/7 | 14/23 | 0.6087 | 2/10 | 2 | 0 | 2 |

Five of ten runtime second-hop queries were valid and executed; two missing gold facts were recovered, in two cases. The other five cases used first-hop fallback. No tested hop-balanced allocation lost a first-hop gold fact in these ten cases.

### Condition B — Offline expected-bridge intervention (4 valid retrieval attempts only)

| Policy | Gold facts retrieved | Recall | Complete cases | Newly recovered facts | Original facts lost |
|---|---:|---:|---:|---:|---:|
| First-hop only | 5/9 | 0.5556 | 0/4 | 0 | 0 |
| Naive rank union | 5/9 | 0.5556 | 0/4 | 0 | 0 |
| Hop-balanced 5/5 | 9/9 | 1.0000 | 4/4 | 4 | 0 |
| Hop-balanced 7/3 | 9/9 | 1.0000 | 4/4 | 4 | 0 |
| Hop-balanced 3/7 | 9/9 | 1.0000 | 4/4 | 4 | 0 |

Condition B excludes four cases with no defined expected bridge and two cases with invalid expected-bridge queries. Its denominator is **four actual retrieval attempts**, not ten cases. It must never be reported as runtime performance.

## Interpretation and limitations

- Naive first-hop-priority concatenation contributes no second-hop chunks when the original top-10 fills the context budget.
- Hop balancing can incorporate newly recovered evidence, but synthetic testing shows it can also displace necessary original evidence.
- All three allocations tied on these cases; there is **no empirical basis to declare 5/5 optimal**.
- The ten cases were chosen with an offline oracle, not selected by the deployed Phase 9 sufficiency detector. Results are diagnostics, not unbiased recovery rates or held-out test results.
- Exact HotpotQA supporting-fact coverage is distinct from practical answer sufficiency; a passage may be answer-useful without matching the exact gold sentence IDs.
- Bridge selection and query formulation remain substantial upstream bottlenecks.

## Decision

Retain **hop-balanced 5/5 as a provisional, non-optimized baseline**, solely for further controlled comparison. Do not claim that it is evidence-preserving in the sense of guaranteeing no lost supporting facts. Do not tune or select the final production policy on these ten diagnostic cases.

## Next experiment

1. Compare against original-question cross-encoder reranking of the deduplicated first-hop + second-hop candidate pool; do not directly compare raw cross-encoder scores produced for different queries.
2. Evaluate on a broader, predeclared validation sample including original evidence at ranks 8–10 and both triggered and untriggered recovery cases as appropriate.
3. Report new/lost gold facts, complete evidence, repaired/harmed cases, and runtime versus offline intervention separately.
4. Freeze a policy only after broader validation; reserve the held-out test set for final evaluation.

**Repository workflow:** No commit or GitHub push until Phase 10 is complete.
