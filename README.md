# AdaptiveRAG

## Failure-Aware Retrieval Diagnosis and Targeted Recovery

AdaptiveRAG is a research-oriented Retrieval-Augmented Generation (RAG) project investigating whether a RAG system can identify **why retrieval failed** and select an appropriate recovery strategy instead of applying the same corrective action to every query.

The project is motivated by a simple observation:

> **Semantic relevance is not the same as evidence sufficiency.**

A retriever can return highly relevant passages while still failing to retrieve all of the evidence required to answer a question correctly.

The central research question is:

> **Does explicit failure diagnosis lead to better recovery decisions than static RAG or generic corrective retrieval?**

The repository is being developed incrementally. Each component is implemented, validated, and analyzed before the next stage is introduced.

> **Current status:** Corpus engineering, evidence-aware sentence chunking, dense retrieval, BM25 sparse retrieval, hybrid Reciprocal Rank Fusion, cross-encoder reranking, supporting-fact retrieval evaluation, and offline oracle failure diagnosis are implemented. Adaptive runtime diagnosis, targeted recovery, post-recovery verification, generation, and abstention remain future work.

---

## Table of Contents

- [Motivation](#motivation)
- [Research Question](#research-question)
- [Core Architecture](#core-architecture)
- [Failure Taxonomy](#failure-taxonomy)
- [Dataset Strategy](#dataset-strategy)
- [HotpotQA Corpus Engineering](#hotpotqa-corpus-engineering)
- [Evidence-Aware Sentence Chunking](#evidence-aware-sentence-chunking)
- [Retrieval Architecture](#retrieval-architecture)
- [Development Evaluation](#development-evaluation)
- [Dense and Sparse Complementarity](#dense-and-sparse-complementarity)
- [Hybrid Candidate-Loss Analysis](#hybrid-candidate-loss-analysis)
- [Cross-Encoder Repair Analysis](#cross-encoder-repair-analysis)
- [Oracle Failure Diagnosis](#oracle-failure-diagnosis)
- [Current Findings](#current-findings)
- [Current Limitations](#current-limitations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Roadmap](#roadmap)
- [Research Engineering Principles](#research-engineering-principles)
- [Author](#author)

---

# Motivation

A conventional RAG system often follows a static pipeline:

```text
User Query
    |
    v
Retrieve Top-K
    |
    v
Construct Context
    |
    v
Generate Answer
```

This assumes that retrieval produced sufficient evidence.

In practice, retrieval can fail in fundamentally different ways:

- the wrong document may be retrieved,
- the correct document may be retrieved but the wrong passage selected,
- only part of a multi-hop evidence chain may be recovered,
- information may be outdated,
- retrieved sources may conflict,
- the query may be ambiguous,
- or generation may introduce claims unsupported by retrieved evidence.

These failures should not necessarily trigger the same intervention.

For example:

```text
Wrong document
    -> query rewriting / retrieval change

Correct document, wrong passage
    -> passage recovery / context expansion

Missing second-hop evidence
    -> query decomposition / multi-hop retrieval

Ambiguous query
    -> clarification

Conflicting evidence
    -> source comparison / verification

Insufficient verified evidence
    -> abstention
```

AdaptiveRAG investigates whether explicitly diagnosing the failure mechanism can support more appropriate recovery decisions.

---

# Research Question

The primary research question is:

> **Does explicit failure diagnosis lead to better recovery decisions than static RAG or generic corrective retrieval?**

The project separates four related problems:

```text
Detection
    |
    | Something appears wrong.
    v
Diagnosis
    |
    | What specifically failed?
    v
Recovery
    |
    | Which intervention should be used?
    v
Verification
    |
    | Did the intervention actually repair the evidence?
    v
Generate / Recover Again / Abstain
```

This separation is important.

Detecting weak evidence does not automatically reveal why the evidence is weak.

Likewise, correctly diagnosing a failure does not guarantee that a chosen recovery action will repair it.

The project therefore evaluates these stages separately.

---

# Core Architecture

The intended end-to-end architecture is:

```text
User Query
    |
    v
Initial Retrieval
    |
    v
Evidence Inspection
    |
    v
Is the evidence sufficient?
    |
    +---------------- Yes ----------------+
    |                                     |
    No                                    v
    |                               Generate Answer
    v
Failure Diagnosis
    |
    v
Recovery Router
    |
    v
Targeted Intervention
    |
    v
Re-retrieval
    |
    v
Post-Recovery Verification
    |
    +------------- Sufficient -----------> Generate
    |
    +------------- Insufficient
                        |
                        v
                Recovery budget?
                   /        \
                 Yes        No
                  |          |
                  v          v
             Recover Again  Abstain
```

The goal is not simply to add more retrieval components.

The goal is to study whether the system can determine:

1. when recovery is necessary,
2. why the current evidence is insufficient,
3. which recovery action is appropriate,
4. and whether that recovery actually fixed the problem.

---

# Failure Taxonomy

The project defines semantic failure states separately from the raw signals used to detect them.

Current failure types include:

```text
SUFFICIENT_EVIDENCE
SEVERE_RETRIEVAL_FAILURE
DOCUMENT_SELECTION_FAILURE
PASSAGE_SELECTION_FAILURE
EVIDENCE_COVERAGE_FAILURE
AMBIGUOUS_QUERY
CONFLICTING_EVIDENCE
TEMPORAL_FAILURE
UNSUPPORTED_INFERENCE
UNKNOWN
```

They are grouped into broader failure families:

```text
Retrieval
Evidence
Generation
None
```

## Retrieval-oriented failures

### Document Selection Failure

The required source document is not successfully retrieved.

Potential recovery:

- query rewriting,
- retrieval-strategy change,
- hybrid retrieval,
- query decomposition,
- metadata filtering.

### Passage Selection Failure

The required source document is reached, but the selected chunk does not contain all required evidence.

Potential recovery:

- neighboring-context expansion,
- alternative chunk selection,
- passage reranking,
- larger evidence windows.

### Evidence Coverage Failure

Some required evidence is retrieved, but the evidence chain remains incomplete.

This is especially important for multi-hop questions.

Potential recovery:

- query decomposition,
- second-hop retrieval,
- evidence-conditioned retrieval,
- additional targeted retrieval.

## Other planned failure classes

The taxonomy also reserves states for:

- ambiguous queries,
- conflicting evidence,
- temporal failures,
- unsupported inference.

These states are defined structurally but are not yet evaluated by the current HotpotQA retrieval oracle.

---

# Dataset Strategy

Different datasets are intended for different experimental purposes rather than being merged into one large benchmark.

## HotpotQA

HotpotQA is the primary development benchmark.

It is useful because questions include annotated supporting facts, enabling evaluation of whether retrieval recovered the evidence associated with the answer.

The project currently uses the `distractor` configuration.

Training split:

```text
90,447 questions
```

Validation split:

```text
7,405 questions
```

The current development experiments use the first 100 training examples and construct a pooled retrieval corpus from their contexts.

This is intentionally a **development sanity slice**, not a final held-out benchmark.

Results from this slice should not be interpreted as final benchmark performance.

## Planned additional evaluation

Future evaluation may include:

- HotpotQA held-out validation queries,
- BEIR retrieval benchmarks,
- CRAG-style reliability experiments,
- RAGBench,
- controlled failure-injection experiments.

These have not yet been implemented.

---

# HotpotQA Corpus Engineering

Before implementing retrieval, the project performs explicit corpus-integrity analysis.

## Raw training corpus

The full HotpotQA training split contains:

```text
QA examples:                 90,447
Document appearances:      899,667
Unique normalized titles:  482,021
```

A document appearance represents a document occurring in one question's context.

The same source can therefore appear multiple times across questions.

## Version-aware document identity

Title-only deduplication was rejected after the corpus audit found:

```text
Titles with multiple content versions: 1,661
```

Therefore:

```text
title -> document
```

is not always a safe one-to-one mapping.

AdaptiveRAG instead constructs document identity from:

```text
normalized title
        +
normalized document content
        |
        v
SHA-256 fingerprint
        |
        v
stable document ID
```

The implementation uses the first 16 hexadecimal characters of the SHA-256 digest.

Document IDs therefore follow the general form:

```text
hotpotqa_<content_fingerprint>
```

This produces:

```text
483,682 unique document versions
```

from:

```text
899,667 document appearances
```

The difference between:

```text
483,682 document versions
```

and:

```text
482,021 normalized titles
```

is:

```text
1,661
```

which matches the number of normalized titles observed with multiple content versions.

This provides an independent consistency check on corpus construction.

---

# Sentence-Level Evidence Preservation

HotpotQA supporting evidence is represented using document titles and sentence positions.

Conceptually:

```text
(title, sentence_id)
```

Because sentence IDs are positional, preprocessing must preserve sentence structure.

Empty source sentence positions are therefore preserved in metadata rather than deleted in a way that would shift later sentence IDs.

Gold evidence is internally represented using:

```text
(document_id, sentence_id)
```

where `document_id` identifies the exact content version associated with the question.

## Corpus integrity findings

The full training-corpus audit found:

```text
Total supporting facts:        215,684
Valid supporting facts:        215,662
Missing supporting documents:        0
Invalid sentence IDs:                22
Gold facts on empty sentences:        0
```

The 22 malformed sentence IDs originate from source benchmark annotations.

The project does **not silently repair benchmark ground truth**.

Malformed annotations are preserved, detected, reported, and handled explicitly by evaluation logic.

---

# Evidence-Aware Sentence Chunking

## Status: Complete

The original character chunker is retained as a generic baseline.

A HotpotQA-compatible sentence-aware chunker was added so retrieved chunks can be mapped exactly back to benchmark evidence.

Current configuration:

```text
Sentences per chunk: 3
Sentence overlap:    1
Step size:           2
```

A sentence-aware chunk preserves metadata such as:

```text
Chunk
|
+-- document_id
+-- chunk_index
+-- title
+-- sentence_ids
+-- text
```

Empty source sentence positions remain structurally preserved while fully empty chunks are skipped.

## Full-corpus validation

The sentence-aware chunker was validated across the full version-aware HotpotQA training corpus:

```text
Documents:                    483,682
Sentence-aware chunks:        821,585
Evidence-index entries:     1,802,257

Valid gold facts mapped:      215,662 / 215,662
Malformed annotations:             22
```

Therefore:

```text
100% of valid annotated supporting facts
```

are structurally representable by the sentence-aware chunks.

This is a **mapping/integrity result**, not retrieval accuracy.

---

# Retrieval Architecture

The current retrieval experiments use the same sentence-aware chunks and exact `(document_id, sentence_id)` evidence representation.

For retrieval, chunk text is represented as:

```text
<title>: <chunk text>
```

The stored raw chunk text remains unchanged.

## Dense Retrieval

Dense retrieval uses:

```text
Model:
sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
384

Normalization:
L2

Index:
FAISS IndexFlatIP
```

Because embeddings are normalized, inner product corresponds to cosine similarity.

The dense baseline uses exact FAISS search for the current development corpus.

## BM25 Sparse Retrieval

A deterministic in-memory BM25 retriever was implemented without an external BM25 package.

Current configuration:

```text
k1 = 1.5
b  = 0.75
```

Tokenization uses lowercasing and a simple word-token regular expression.

No stemming or stopword removal is currently applied.

BM25 and dense retrieval use the same title-plus-chunk retrieval representation.

## Hybrid Retrieval

Dense and sparse retrieval are combined using Reciprocal Rank Fusion.

Current configuration:

```text
Dense candidate depth:   20
BM25 candidate depth:    20
RRF k:                    60
Final evaluation depth:  10
```

For a chunk appearing at rank `r`:

```text
RRF contribution = 1 / (60 + r)
```

Contributions from dense and sparse retrieval are summed.

Raw BM25 and cosine scores are not directly combined because they are not on the same scale.

## Cross-Encoder Reranking

The current strongest ranking baseline uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Pipeline:

```text
Query
  |
  +--> Dense@20 ----+
  |                 |
  +--> BM25@20 -----+
                    |
                    v
             Deduplicate Candidates
                    |
                    v
              Cross-Encoder
                    |
                    v
                 Top 10
```

The cross-encoder jointly scores `(query, passage)` pairs.

Its output is treated as a relevance score, **not a calibrated probability or confidence value**.

It is also not treated as an evidence-sufficiency model.

---

# Development Evaluation

## Experimental setup

The current development evaluation uses:

```text
HotpotQA distractor training examples: 100
Unique pooled documents:               992
Sentence-aware chunks:               1,911
Valid gold supporting facts:           249
```

These experiments are intended for architecture validation and failure analysis.

They are **not final held-out benchmark results** and have not been used to make general performance claims.

## Metrics

### Evidence Recall@K

Fraction of individual valid gold supporting facts represented in the retrieved chunks.

### Question Hit@K

Fraction of questions where at least one gold supporting fact is retrieved.

### Complete Evidence@K

Fraction of questions where all valid annotated supporting facts are represented.

### MRR@10

Mean reciprocal rank of the first retrieved chunk containing any gold evidence.

Complete evidence is deliberately separated from question hit rate.

A system may retrieve one useful fact while still missing another fact required for a multi-hop evidence chain.

---

# Retrieval Results

## Dense MiniLM

```text
@1
Evidence Recall:       0.3896
Question Hit:          0.8500
Complete Evidence:     0.0000

@5
Evidence Recall:       0.7149
Question Hit:          0.9800
Complete Evidence:     0.4700

@10
Evidence Recall:       0.8153
Question Hit:          0.9900
Complete Evidence:     0.6400

MRR@10:                0.9000
```

At `K=10`, dense retrieval recovered:

```text
203 / 249 gold facts
64 / 100 complete evidence sets
```

A notable gap appears between:

```text
Question Hit@10:       99%
Complete Evidence@10: 64%
```

Nearly every question receives some evidence, but substantially fewer receive the complete annotated evidence set.

---

## BM25

```text
@1
Evidence Recall:       0.3976
Question Hit:          0.8000
Complete Evidence:     0.0000

@5
Evidence Recall:       0.7430
Question Hit:          0.9700
Complete Evidence:     0.5000

@10
Evidence Recall:       0.8554
Question Hit:          0.9900
Complete Evidence:     0.7000

MRR@10:                0.8707
```

At `K=10`, BM25 recovered:

```text
213 / 249 gold facts
70 / 100 complete evidence sets
```

On this development slice, BM25 produced greater evidence coverage at depth 10 while dense retrieval produced a higher MRR.

This is treated as evidence of complementary retrieval behavior rather than a general claim that one retrieval paradigm is superior.

---

## Hybrid RRF

```text
@1
Evidence Recall:       0.4016
Question Hit:          0.8500
Complete Evidence:     0.0000

@5
Evidence Recall:       0.7631
Question Hit:          0.9900
Complete Evidence:     0.5500

@10
Evidence Recall:       0.8394
Question Hit:          1.0000
Complete Evidence:     0.6800

MRR@10:                0.9126
```

At `K=10`, RRF recovered:

```text
209 / 249 gold facts
68 / 100 complete evidence sets
```

RRF improved several early-ranking metrics but did not fully convert dense/sparse complementarity into complete evidence coverage.

---

## Cross-Encoder Reranking

```text
@1
Evidence Recall:       0.4297
Question Hit:          0.9200
Complete Evidence:     0.0000

@5
Evidence Recall:       0.7711
Question Hit:          0.9900
Complete Evidence:     0.5400

@10
Evidence Recall:       0.8916
Question Hit:          1.0000
Complete Evidence:     0.7500

MRR@10:                0.9533
```

At `K=10`, the cross-encoder recovered:

```text
222 / 249 gold facts
75 / 100 complete evidence sets
```

Among the configurations tested on this development slice, the cross-encoder produced the highest Evidence Recall@10, Complete Evidence@10, and MRR@10.

Again, these are development results rather than final held-out benchmark results.

---

# Baseline Comparison

| Retrieval configuration | Evidence Recall@10 | Question Hit@10 | Complete Evidence@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Dense MiniLM | 0.8153 | 0.9900 | 0.6400 | 0.9000 |
| BM25 | 0.8554 | 0.9900 | 0.7000 | 0.8707 |
| Hybrid RRF | 0.8394 | 1.0000 | 0.6800 | 0.9126 |
| Cross-Encoder | 0.8916 | 1.0000 | 0.7500 | 0.9533 |

These results demonstrate why retrieval quality should not be summarized using a single metric.

For example:

- BM25 retrieved more total gold evidence than dense retrieval at depth 10.
- Dense retrieval ranked the first useful evidence somewhat earlier according to MRR.
- RRF improved early ranking but lost some evidence available to individual retrievers.
- Cross-encoder reranking recovered many of those ranking losses but did not eliminate them.

---

# Dense and Sparse Complementarity

The exact gold-evidence overlap between Dense@10 and BM25@10 was analyzed.

## Fact-level overlap

```text
Total gold facts:     249

Recovered by both:    190
Dense only:            13
BM25 only:              23
Neither:                23
```

Therefore:

```text
Dense Recall@10:       203 / 249 = 0.8153
BM25 Recall@10:        213 / 249 = 0.8554
Oracle union:          226 / 249 = 0.9076
```

The oracle union is an **offline evidence-availability analysis**, not a deployable retrieval result.

## Question-level complete evidence

```text
Complete by both:              56
Complete only by Dense:         8
Complete only by BM25:         14
Complete by neither alone:     22

Dense complete:                64%
BM25 complete:                 70%
Oracle union complete:         79%
```

The overlap analysis shows genuine complementarity between lexical and semantic retrieval.

However, evidence available across two ranked lists still has to be selected effectively for the final context.

---

# Hybrid Candidate-Loss Analysis

To distinguish candidate-generation failure from final-ranking failure, the actual RRF candidate pool was analyzed:

```text
Dense@20 ∪ BM25@20
        |
        v
RRF
        |
        v
Final Top 10
```

## Fact-level availability

```text
Total gold facts:                          249

Available in Dense@20 ∪ BM25@20:          236
Candidate-pool evidence availability:   94.78%

Retained by RRF@10:                        209
RRF Evidence Recall@10:                 83.94%

Candidate-available facts lost by RRF:      27
Facts unavailable from both candidates:     13
```

This reveals two different mechanisms:

```text
Candidate-generation failure
    Required evidence never enters
    the candidate pool.

Ranking/fusion failure
    Required evidence enters the
    candidate pool but is removed
    from the final top-k.
```

## Question-level attribution

Among the 32 RRF-incomplete questions:

```text
Pure ranking/fusion loss:       21  (65.6%)
Pure candidate failure:         10  (31.2%)
Mixed failure:                   1   (3.1%)
```

This motivated testing a stronger relevance reranker before implementing adaptive recovery.

---

# Cross-Encoder Repair Analysis

Cross-encoder reranking was applied to the same Dense@20 + BM25@20 candidate pool.

This allows a controlled comparison:

```text
Same candidate generation
        |
        +--> RRF ----------> Top 10
        |
        +--> CrossEncoder -> Top 10
```

## Question-level transitions

```text
RRF complete   -> CE complete:    61
RRF incomplete -> CE complete:    14
RRF complete   -> CE incomplete:   7
RRF incomplete -> CE incomplete:  18
```

Therefore, the change from:

```text
RRF complete: 68
CE complete:  75
```

is not simply seven repaired questions.

The cross-encoder:

```text
repaired 14 previously incomplete questions
introduced 7 complete-to-incomplete regressions
-----------------------------------------------
net improvement: +7 complete questions
```

This is important because aggregate metrics alone would hide those regressions.

## Pure ranking-loss subset

Phase 7 identified:

```text
21
```

questions where all required gold evidence existed in the candidate pool but RRF failed to retain a complete evidence set.

Cross-encoder results on those questions:

```text
Repaired to complete:     14
Still incomplete:          7

Question repair rate:  66.7%
```

## Fact-level repair

RRF@10 was missing:

```text
40 gold facts
```

Of those:

```text
27 were available in the candidate pool
13 were unavailable from the candidate pool
```

Among the 27 candidate-available facts lost by RRF:

```text
Recovered by Cross-Encoder:       20
Still missing after Cross-Encoder: 7
```

The cross-encoder therefore recovered many ranking losses, but stronger pointwise relevance scoring did not completely solve evidence selection.

---

# Why Relevance Is Not Evidence Sufficiency

The cross-encoder scores individual query-passage pairs.

Conceptually:

```text
relevance(query, passage_i)
```

But multi-hop evidence completeness is closer to a set-level objective:

```text
sufficiency(
    query,
    {passage_1, passage_2, ..., passage_k}
)
```

A passage can receive a lower individual relevance score while still containing the missing fact required to complete an evidence chain.

This distinction appears in the development experiments.

Some questions have all required evidence available in the candidate pool but remain incomplete after both RRF and cross-encoder reranking.

This motivates explicit evidence-sufficiency reasoning rather than assuming that progressively stronger relevance ranking alone will solve every retrieval failure.

---

# Oracle Failure Diagnosis

The project separates:

```text
Offline Oracle Diagnosis
```

from:

```text
Runtime Predicted Diagnosis
```

## Oracle diagnosis

During benchmark analysis, HotpotQA gold evidence can determine exactly which required facts and documents were retrieved.

This allows deterministic offline failure attribution.

For example:

```text
Complete gold evidence
    -> SUFFICIENT_EVIDENCE

No gold evidence
    -> SEVERE_RETRIEVAL_FAILURE

All gold documents reached
but required sentence missing
    -> PASSAGE_SELECTION_FAILURE

Partial evidence +
missing required gold document
    -> EVIDENCE_COVERAGE_FAILURE
```

These labels use benchmark ground truth.

They are therefore **not available to a deployed RAG system**.

## Runtime diagnosis

A future runtime diagnoser must infer failure states without gold evidence.

Potential observable signals include:

- retrieval scores,
- score margins,
- dense/sparse agreement,
- evidence redundancy,
- source diversity,
- entity coverage,
- entailment signals,
- contradiction signals,
- evidence-sufficiency judgments,
- metadata and temporal signals.

The long-term evaluation will compare:

```text
Predicted Diagnosis
        |
        v
Oracle Diagnosis
        |
        v
Diagnosis Accuracy
```

---

# Residual Oracle Failures After Cross-Encoder Reranking

The original Dense@10 oracle distribution was:

```text
Sufficient evidence:              64
Severe retrieval failure:          1
Document selection failure:        0
Passage selection failure:         5
Evidence coverage failure:        30
```

After Dense@20 + BM25@20 candidate generation and cross-encoder reranking:

```text
Sufficient evidence:              75
Severe retrieval failure:          0
Document selection failure:        0
Passage selection failure:         4
Evidence coverage failure:        21
```

Therefore:

```text
25 residual incomplete questions
|
+-- 21 evidence-coverage failures
|
+-- 4 passage-selection failures
```

Among the residual incomplete questions:

```text
84% are evidence-coverage failures
16% are passage-selection failures
```

The dominant residual problem is therefore not complete retrieval collapse.

Instead, the current system usually retrieves **part of the required evidence chain while missing another required source or evidence component**.

This observation motivates the next research stage: targeted recovery for incomplete evidence.

---

# Current Findings

The current development experiments support several observations.

## 1. Question hit rate can hide incomplete evidence

Dense retrieval reached:

```text
Question Hit@10:       99%
Complete Evidence@10: 64%
```

Retrieving at least one useful fact does not imply that the evidence set is complete.

## 2. Dense and sparse retrieval are complementary

At `K=10`:

```text
Dense-only gold facts: 13
BM25-only gold facts:  23
```

Neither retrieval strategy strictly subsumes the other on this development slice.

## 3. Candidate generation and final ranking are different failure points

Dense@20 ∪ BM25@20 contained:

```text
236 / 249 = 94.78%
```

of gold evidence facts.

RRF@10 retained only:

```text
209 / 249 = 83.94%
```

Candidate availability therefore does not guarantee final evidence coverage.

## 4. Stronger relevance reranking repairs many ranking failures

The cross-encoder repaired:

```text
14 / 21 = 66.7%
```

of pure RRF ranking-loss questions.

It recovered:

```text
20 / 27
```

candidate-available gold facts that RRF had lost.

## 5. Stronger reranking can also introduce regressions

Cross-encoder reranking repaired 14 RRF-incomplete questions but caused 7 RRF-complete questions to become incomplete.

A globally stronger aggregate metric therefore does not imply uniformly better evidence selection for every query.

## 6. Residual failures are dominated by incomplete evidence coverage

After cross-encoder reranking:

```text
21 / 25
```

residual incomplete questions are oracle-classified as evidence-coverage failures.

This provides empirical motivation for failure-aware retrieval recovery.

---

# Current Limitations

The current implementation intentionally has several limitations.

## Development slice

Current retrieval experiments use the first 100 HotpotQA training examples.

They are development experiments, not final held-out benchmark results.

## Benchmark-derived corpus

The retrieval corpus is pooled from HotpotQA contexts.

It should be described as a **pooled benchmark-derived retrieval corpus**, not full open-Wikipedia retrieval.

## Oracle diagnosis uses gold labels

Current failure attribution relies on exact benchmark supporting facts.

A deployed system will not have these labels.

Runtime failure prediction remains future work.

## Cross-encoder is a relevance reranker

The current cross-encoder was trained for passage relevance.

It is not an evidence-sufficiency model and does not directly optimize multi-passage evidence completeness.

## Exact dense search

The current development corpus uses FAISS `IndexFlatIP`.

Scaling to the full sentence-chunk corpus will require careful consideration of indexing, persistence, memory, and potentially approximate nearest-neighbor search.

## Answer generation is not yet evaluated

The current experiments evaluate retrieval and annotated evidence coverage.

They do not yet establish:

- answer accuracy,
- faithfulness improvement,
- hallucination reduction,
- citation correctness,
- recovery success,
- or abstention quality.

## Gold completeness is not identical to answer sufficiency

HotpotQA annotated supporting facts provide a useful benchmark signal, but retrieving every annotated supporting fact is not always equivalent to the minimum evidence required to answer a question.

Future evaluation should therefore distinguish:

```text
benchmark evidence completeness
```

from:

```text
answer sufficiency
```

and:

```text
answer correctness / faithfulness
```

---

# Repository Structure

Current core structure:

```text
adaptive-rag/
|
+-- data/
|   |
|   +-- demo/
|
+-- scripts/
|   |
|   +-- audit_hotpotqa_corpus.py
|   +-- inspect_hotpotqa.py
|   +-- inspect_hotpotqa_duplicate.py
|   |
|   +-- test_ingestion.py
|   +-- test_chunking.py
|   +-- test_hotpotqa_adapter.py
|   |
|   +-- test_sentence_chunking.py
|   +-- test_sentence_chunking_edge_cases.py
|   +-- validate_sentence_chunking_full.py
|   +-- trace_hotpotqa_text.py
|   |
|   +-- test_dense_embeddings.py
|   +-- test_dense_retrieval.py
|   +-- evaluate_dense_retrieval.py
|   +-- analyze_dense_failures.py
|   |
|   +-- test_sparse_retrieval.py
|   +-- evaluate_sparse_retrieval.py
|   +-- analyze_retrieval_complementarity.py
|   |
|   +-- test_hybrid_retrieval.py
|   +-- evaluate_hybrid_retrieval.py
|   +-- analyze_hybrid_candidate_loss.py
|   |
|   +-- test_cross_encoder_reranking.py
|   +-- evaluate_cross_encoder_reranking.py
|   +-- analyze_reranker_repairs.py
|   |
|   +-- test_oracle_diagnosis.py
|   +-- analyze_cross_encoder_oracle_diagnosis.py
|
+-- src/
|   |
|   +-- chunking/
|   |   +-- chunk.py
|   |   +-- chunker.py
|   |   +-- sentence_chunker.py
|   |
|   +-- diagnosis/
|   |   +-- states.py
|   |   +-- oracle.py
|   |
|   +-- embeddings/
|   |   +-- dense_embedder.py
|   |
|   +-- ingestion/
|   |   +-- document.py
|   |   +-- hotpotqa.py
|   |   +-- loader.py
|   |
|   +-- retrieval/
|       +-- dense_retriever.py
|       +-- sparse_retriever.py
|       +-- hybrid_retriever.py
|       +-- reranker.py
|
+-- .gitignore
+-- README.md
+-- requirements.txt
```

Generated embeddings, retrieval indexes, datasets, model artifacts, caches, and experiment outputs are excluded from version control.

---

# Installation

## Environment

The project has been developed with:

```text
Python 3.11
VS Code
Windows PowerShell
```

## Clone

```bash
git clone https://github.com/Sahithi102302/adaptive-rag.git
cd adaptive-rag
```

## Create virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
pip install -r requirements.txt
```

Current validated dependencies:

```text
datasets==5.0.1
pyarrow==21.0.0
pandas==3.0.6
numpy==2.4.6
sentence-transformers==6.1.0
scikit-learn==1.9.1
faiss-cpu==1.15.1
```

`pyarrow==21.0.0` is pinned because it has been validated with the current Hugging Face `datasets` environment used by this project.

---

# Running the Pipeline

Commands should be run from the repository root with the virtual environment activated.

## Corpus and ingestion

```powershell
python -m scripts.test_ingestion
python -m scripts.test_chunking
python -m scripts.inspect_hotpotqa
python -m scripts.test_hotpotqa_adapter
python -m scripts.audit_hotpotqa_corpus
python -m scripts.validate_hotpotqa_full
```

## Sentence-aware chunking

```powershell
python -m scripts.test_sentence_chunking
python -m scripts.test_sentence_chunking_edge_cases
python -m scripts.validate_sentence_chunking_full
```

## Dense retrieval

```powershell
python -m scripts.test_dense_embeddings
python -m scripts.test_dense_retrieval
python -m scripts.evaluate_dense_retrieval
python -m scripts.analyze_dense_failures
```

## BM25 retrieval

```powershell
python -m scripts.test_sparse_retrieval
python -m scripts.evaluate_sparse_retrieval
```

## Dense/sparse complementarity

```powershell
python -m scripts.analyze_retrieval_complementarity
```

## Hybrid RRF

```powershell
python -m scripts.test_hybrid_retrieval
python -m scripts.evaluate_hybrid_retrieval
python -m scripts.analyze_hybrid_candidate_loss
```

## Cross-encoder reranking

```powershell
python -m scripts.test_cross_encoder_reranking
python -m scripts.evaluate_cross_encoder_reranking
python -m scripts.analyze_reranker_repairs
```

## Oracle diagnosis

```powershell
python -m scripts.test_oracle_diagnosis
python -m scripts.analyze_cross_encoder_oracle_diagnosis
```

Some scripts download HotpotQA or Hugging Face models on first execution.

---

# Roadmap

## Completed

### Corpus foundation

- [x] Project research framing
- [x] Generic document abstraction
- [x] TXT and JSON ingestion
- [x] Character chunking baseline
- [x] HotpotQA integration
- [x] Full raw-corpus audit
- [x] Version-aware document identity
- [x] Corpus deduplication
- [x] Exact supporting-fact mapping
- [x] Full adapter validation

### Evidence-aware chunking

- [x] Sentence-aware chunk representation
- [x] Sentence-position preservation
- [x] Evidence-to-chunk mapping
- [x] Full-corpus evidence-mapping validation

### Retrieval baselines

- [x] Dense embedding pipeline
- [x] FAISS dense retrieval
- [x] Dense retrieval evaluation
- [x] BM25 sparse retrieval
- [x] Dense/sparse complementarity analysis
- [x] Hybrid Reciprocal Rank Fusion
- [x] Candidate-loss analysis
- [x] Cross-encoder reranking
- [x] Reranker repair/regression analysis

### Failure diagnosis foundation

- [x] Failure-state representation
- [x] Diagnostic-signal representation
- [x] Offline oracle retrieval diagnosis
- [x] Dense baseline failure analysis
- [x] Cross-encoder residual failure analysis

## Next

### Runtime evidence sufficiency and diagnosis

- [ ] Define inference-time observable diagnostic signals
- [ ] Implement evidence-sufficiency detector
- [ ] Implement predicted failure diagnosis
- [ ] Compare predicted diagnoses with oracle labels
- [ ] Measure diagnosis accuracy by failure type

### Targeted recovery

- [ ] Recovery router
- [ ] Query rewriting
- [ ] Query decomposition
- [ ] Multi-hop / missing-evidence recovery
- [ ] Passage/context expansion
- [ ] Recovery verification
- [ ] Recovery budget
- [ ] Abstention

### Generation

- [ ] Context construction
- [ ] Baseline answer generation
- [ ] Citation-aware generation
- [ ] Faithfulness evaluation

### Evaluation

- [ ] Recovery success rate
- [ ] Unnecessary recovery rate
- [ ] Answer correctness
- [ ] Faithfulness
- [ ] Citation correctness
- [ ] Latency
- [ ] Computational/API cost
- [ ] Performance by failure type
- [ ] Held-out HotpotQA evaluation

### Additional benchmarks

- [ ] BEIR
- [ ] CRAG-oriented reliability evaluation
- [ ] RAGBench
- [ ] Controlled failure benchmark

---

# Research Engineering Principles

## 1. Validate Before Scaling

Components are first tested on small subsets before full-corpus or larger-scale execution.

## 2. Preserve Ground Truth

Benchmark annotations are not silently changed simply because they appear malformed.

## 3. Separate Relevance from Evidence Sufficiency

A semantically relevant passage is not automatically sufficient evidence.

## 4. Separate Candidate Generation from Ranking

Failure to retrieve evidence into the candidate pool is different from retrieving the evidence and ranking it out of the final context.

## 5. Separate Oracle and Runtime Diagnosis

Gold benchmark evidence may be used for offline analysis, but runtime diagnostic systems must operate without access to those labels.

## 6. Compare Against Strong Baselines

Failure-aware recovery should be compared against progressively stronger static retrieval and reranking systems.

## 7. Analyze Regressions, Not Only Aggregate Gains

A method that improves aggregate performance may still make individual queries worse.

Repair and regression transitions are therefore analyzed explicitly.

## 8. Measure Recovery, Not Just Detection

Correctly identifying a failure is useful only if the selected intervention actually improves the evidence or final answer.

## 9. Measure Cost as Well as Quality

Adaptive recovery introduces additional computation.

Latency, model calls, token usage, and computational/API cost should therefore be measured alongside quality.

## 10. Keep Implemented and Planned Work Distinct

The repository distinguishes between:

- implemented components,
- validated development results,
- offline oracle analyses,
- planned experiments,
- and research hypotheses.

Future functionality is not presented as completed work.

---

# Current Project Status

The project has completed the retrieval-baseline and reranking foundation required to begin studying failure-aware recovery.

The strongest current tested development pipeline is:

```text
Query
  |
  +--> Dense Retrieval @20
  |
  +--> BM25 Retrieval @20
            |
            v
      Candidate Union
            |
            v
       Cross-Encoder
            |
            v
          Top 10
            |
            v
   Oracle Failure Analysis
```

On the current 100-question development slice:

```text
Evidence Recall@10:       89.16%
Question Hit@10:         100.00%
Complete Evidence@10:     75.00%
MRR@10:                    0.9533
```

The remaining incomplete cases are dominated by:

```text
Evidence coverage failure: 21
Passage selection failure:  4
```

The next research stage is to move from **offline oracle diagnosis** toward **runtime evidence-sufficiency detection and predicted failure diagnosis**, followed by targeted recovery experiments.

No claims are currently made that adaptive recovery improves answer quality, faithfulness, latency, hallucination rate, or cost because those experiments have not yet been implemented.


---

# Phase 10 — Failure-Aware Retrieval Recovery

## Status: Implementation Complete — Post-Recovery Verification Pending

Phase 10 extends AdaptiveRAG beyond retrieval and failure detection by introducing recovery strategies for questions whose retrieved evidence is predicted to be insufficient.

The central objective is:

> **Can a RAG system recover missing evidence through targeted retrieval interventions while preserving useful original evidence and avoiding unnecessary regressions?**

Earlier phases established that retrieving semantically relevant passages does not guarantee complete evidence coverage.

Phase 9 introduced a runtime evidence-sufficiency detector that predicts whether the retrieved evidence is sufficient without accessing HotpotQA ground-truth annotations.

Phase 10 builds on this detector to investigate:

1. which retrieval interventions can recover missing evidence,
2. how multi-hop retrieval can discover additional evidence,
3. how original and recovered evidence should be combined,
4. when the system should attempt recovery,
5. and how recovery candidates can be prepared safely for subsequent verification.

The current implementation supports recovery execution, evidence selection, runtime controller routing, and recovery-candidate handoff.

**Independent post-recovery verification is not yet implemented.** Recovery execution and successful evidence repair are therefore treated as separate outcomes.

---

## Phase 10 Architecture

The implemented recovery architecture is organized as follows:

```text
User Query
    |
    v
Initial Retrieval
    |
    +--> Dense Retrieval
    |
    +--> BM25 Retrieval
              |
              v
        Candidate Union
              |
              v
         Cross-Encoder
              |
              v
        Original Top 10
              |
              v
     Runtime Feature Extraction
              |
              v
      Sufficiency Detection
              |
              v
        Recovery Controller
              |
       +------+------+
       |             |
       v             v
   Keep Original   Attempt Recovery
       |             |
       |             v
       |       Recovery Intervention
       |             |
       |             v
       |       Recovered Evidence
       |             |
       |             v
       |       Evidence Selection
       |             |
       |             v
       |       Recovery Candidate
       |             |
       |             v
       |       Pending Verification
       |             |
       +-------------+
              |
              v
       Phase 11 Verification
           (Planned)
```

A central design principle is that **recovered evidence is not automatically accepted as sufficient evidence**.

The original retrieval results remain available throughout recovery so the system can reject an unsuccessful candidate without losing the initial evidence.

---

## Phase 10.1 — Recovery State Representation

### Status: Complete

The first stage defines explicit recovery contracts so different interventions can be executed through a common interface.

The implementation is located in:

```text
src/recovery/states.py
```

The recovery system defines the following actions:

```text
NONE
EXPANDED_CANDIDATE_RETRIEVAL
QUERY_REFORMULATION
MULTI_QUERY_RETRIEVAL
```

Recovery execution also has explicit statuses:

```text
NOT_ATTEMPTED
COMPLETED
FAILED
```

The primary objects are:

```text
RecoveryRequest
    |
    +-- original_query
    +-- action
    +-- original_results
    +-- metadata

RecoveryResult
    |
    +-- action
    +-- status
    +-- original_query
    +-- recovery_queries
    +-- original_results
    +-- recovered_results
    +-- error_message
    +-- metadata
```

This representation separates the recovery request from the evidence produced by the intervention.

Importantly:

```text
RecoveryStatus.COMPLETED
```

means that the intervention executed successfully.

It does **not** mean that the intervention recovered all required evidence.

This distinction is necessary because retrieval success, evidence completeness, and answer correctness are different evaluation objectives.

---

## Phase 10.2 — Expanded Candidate Retrieval

### Status: Complete

The first recovery intervention investigates whether incomplete evidence can be recovered by expanding the retrieval candidate pool.

The original retrieval baseline uses:

```text
Dense@20
    +
BM25@20
    |
    v
Candidate Union
    |
    v
Cross-Encoder
    |
    v
Top 10
```

Expanded candidate retrieval increases the candidate depth:

```text
Dense@100
    +
BM25@100
    |
    v
Expanded Candidate Union
    |
    v
Cross-Encoder
    |
    v
Top 10
```

The purpose is to distinguish two failure mechanisms:

```text
Candidate-generation failure
    |
    | Required evidence never entered
    | the retrieval candidate pool.
    v
Expand retrieval depth
```

and:

```text
Final-ranking failure
    |
    | Required evidence entered the pool
    | but was excluded from the final Top 10.
    v
Improve evidence selection
```

### Experimental finding

Expanded candidate retrieval produced modest improvements on the tested validation examples.

However, increasing candidate depth did not consistently recover all missing evidence.

Some required facts were present in the expanded candidate pool but were still excluded from the final evidence set.

This indicates that candidate expansion alone is not sufficient.

A recovery system must also consider **how evidence is selected for the final context**.

Implementation:

```text
src/recovery/expanded_candidates.py
```

Related evaluation:

```powershell
python -m scripts.test_expanded_candidate_recovery
python -m scripts.evaluate_expanded_candidate_recovery
python -m scripts.analyze_expanded_candidate_residuals
```

---

## Phase 10.3 — Evidence-Conditioned Multi-Hop Recovery

### Status: Complete

The next intervention investigates whether a second retrieval step can recover evidence missing from the original retrieval results.

This is particularly important for HotpotQA questions that require connecting information across multiple documents.

Conceptually:

```text
Original Question
       |
       v
First-Hop Retrieval
       |
       v
Inspect Retrieved Evidence
       |
       v
Extract Candidate Entities
       |
       v
Select Potential Bridge
       |
       v
Generate Relation Query
       |
       v
Second-Hop Retrieval
       |
       v
Recovered Evidence
```

The system uses the retrieved passages as evidence for constructing a second retrieval query.

The recovery process must operate without knowing which entity or document is correct according to HotpotQA supporting-fact annotations.

### Entity extraction

The current implementation uses:

```text
dslim/bert-base-NER
```

Named entities are extracted from the first-hop retrieved passages.

The experiments also investigate:

- candidate aggregation,
- entity deduplication,
- passage-balanced candidate selection,
- and bridge-entity selection.

The current recovery configuration inspects the first three retrieved passages and limits the number of entity candidates considered from each source.

This prevents a single passage with many recognized entities from dominating candidate selection.

### Bridge selection

A bridge entity is a potential connection between the original question, the first-hop evidence, and another relevant source.

For example:

```text
Question
   |
   v
Document A
   |
   | Mentions Entity B
   v
Bridge Entity B
   |
   v
Document B
   |
   v
Missing Evidence
```

The bridge is selected using observable query and passage information.

Gold supporting documents are not supplied to the runtime bridge selector.

### Relation-query generation

The current implementation experiments with:

```text
Qwen2.5-1.5B-Instruct
```

The model generates a constrained second-hop retrieval query from the original question and the selected bridge candidate.

The implementation uses CPU inference for local experimentation.

The generated query is checked for structural validity before being used for retrieval.

A structurally valid query is not necessarily semantically correct.

### Experimental findings

Initial real-retrieval experiments on 10 oracle-selected incomplete validation questions showed:

```text
Questions evaluated:          10
Complete evidence repairs:     2
```

An offline expected-bridge diagnostic recovered four of six eligible cases.

The expected-bridge experiment uses benchmark knowledge for diagnostic analysis and is **not a deployable runtime result**.

These experiments revealed that second-hop recovery is sensitive to:

- entity extraction quality,
- bridge selection,
- relation-query generation,
- and final evidence selection.

The broader validation experiments subsequently found that generated bridge queries were structurally valid in only 26 of 50 evaluated cases.

This limits the reliability of the current second-hop intervention.

Implementation includes:

```text
src/recovery/entity_extraction.py
src/recovery/bridge_selection.py
src/recovery/llm_bridge_selection.py
src/recovery/bridge_query.py
src/recovery/relation_query.py
src/recovery/qwen_generator.py
src/recovery/multi_query.py
src/recovery/second_hop.py
```

Representative tests:

```powershell
python -m scripts.test_entity_extraction
python -m scripts.test_bridge_selection
python -m scripts.test_bridge_query
python -m scripts.test_relation_query
python -m scripts.test_second_hop
```

The goal of this stage is not to assume that language-model-generated queries are reliable.

Instead, the project evaluates whether evidence-conditioned retrieval produces additional useful evidence and identifies where the recovery process fails.

---

## Phase 10.4 — Evidence-Preserving Selection

### Status: Complete

Second-hop retrieval introduces a new problem.

The system now has two evidence lists:

```text
Original Evidence
       +
Recovered Evidence
```

Simply combining these lists does not guarantee better evidence coverage.

The final context has a limited number of passages.

Adding new evidence can displace useful original evidence.

This creates a potential recovery regression:

```text
Original Top 10
    |
    | Contains useful supporting fact
    v
Recovery
    |
    v
New Top 10
    |
    | Original supporting fact removed
    v
Evidence Regression
```

To investigate this problem, Phase 10 introduces an evidence-preserving selection strategy.

Implementation:

```text
src/recovery/evidence_selection.py
```

### Balanced evidence selection

The primary experimental configuration uses:

```text
Final evidence budget:      10

Reserved first-hop slots:    5
Reserved second-hop slots:   5
```

Conceptually:

```text
First-Hop Evidence
       |
       +---- Top 5 ----+
                     |
                     v
                Final Top 10
                     ^
                     |
       +---- Top 5 ----+
       |
Second-Hop Evidence
```

The selector also:

- deduplicates repeated chunks,
- tracks first-hop and second-hop ranks,
- identifies chunks appearing in both lists,
- fills unused slots when one list has fewer candidates,
- and preserves evidence-source provenance.

The provenance states are:

```text
FIRST_HOP
SECOND_HOP
BOTH
```

Raw retrieval scores from different queries are not directly compared because they are not necessarily on a common scale.

### Phase 10.4E — Broader evidence-selection evaluation

A broader diagnostic experiment evaluated 50 HotpotQA validation questions.

Recovery was deliberately attempted on the evaluation examples to study evidence-selection behavior.

**This is a forced-recovery experiment, not an evaluation of the runtime detector's routing policy.**

The following results were observed:

| Evidence-selection policy | Evidence Recall@10 | Complete Evidence@10 |
|---|---:|---:|
| Original first-hop Top 10 | 0.8305 | 33 / 50 |
| Naive union selection | 0.8305 | 33 / 50 |
| Balanced 5/5 | 0.8729 | 38 / 50 |
| Balanced 7/3 | 0.8559 | 36 / 50 |
| Balanced 3/7 | 0.8729 | 38 / 50 |
| Combined-pool cross-encoder | 0.8559 | 34 / 50 |

The balanced 5/5 policy produced:

```text
Original complete questions:       33 / 50

Balanced 5/5 complete questions:   38 / 50

Incomplete-to-complete repairs:     5

Complete-to-incomplete losses:      0

Net improvement:                   +5
```

The balanced 3/7 policy achieved the same aggregate evidence recall and complete-evidence count on this sample.

Therefore, the experiment does not establish that 5/5 is universally superior.

The 5/5 configuration was retained as the current balanced research baseline.

### Interpretation

These results demonstrate that evidence-selection policy can materially affect the outcome of recovery.

The balanced strategies recovered additional annotated supporting facts that were not present in the original final context.

However, the absence of complete-to-incomplete regressions in 50 examples does not guarantee that future recovery candidates cannot lose useful evidence.

In particular, reserving first-hop slots does not preserve every original Top-10 passage.

This motivates an independent post-recovery verification stage.

Experiment documentation:

```text
docs/experiments/phase_10_4_evidence_selection.md
docs/experiments/phase_10_4e_results.csv
```

Evaluation command:

```powershell
python -m scripts.evaluate_broad_evidence_selection
```

---

## Phase 10.5 — Runtime Recovery Controller

### Status: Complete

Earlier Phase 10 experiments investigated whether recovery interventions could improve evidence coverage.

However, a deployable recovery system cannot use HotpotQA gold labels to decide which questions require recovery.

Phase 10.5 connects the frozen Phase 9 runtime detector to the recovery controller.

The resulting architecture is:

```text
Original Query
      |
      v
Initial Retrieval
      |
      v
Original Top 10
      |
      v
Runtime Feature Extraction
      |
      v
Frozen Sufficiency Detector
      |
      v
Recovery Recommendation
      |
      v
Recovery Controller
      |
      +--> Keep Original
      |
      +--> Attempt Recovery
```

The controller uses the predicted sufficiency result rather than an oracle failure label.

### Phase 9 detector used by the controller

The frozen detector uses:

```text
Model:
Balanced Logistic Regression

Preprocessing:
Median Imputation
Standard Scaling

Feature count:
29

Retrieval-derived features:
19

Lexical evidence features:
10

Sufficiency threshold:
0.58
```

The detector predicts whether the retrieved evidence is likely to be sufficient.

It does not provide reliable fine-grained runtime attribution of every failure type.

Therefore, the current controller primarily distinguishes:

```text
Predicted sufficient
        |
        v
Keep original evidence
```

from:

```text
Predicted insufficient
        |
        v
Attempt recovery
```

The detector threshold is frozen and is not retuned on the held-out test set.

### Held-out detector evaluation

The Phase 9 detector was evaluated on 500 held-out test questions.

```text
Test questions:                    500

Accuracy:                         0.6640

Balanced accuracy:                0.6867

Insufficient-class precision:     0.4217

Insufficient-class recall:        0.7348

Insufficient-class F1:            0.5359

Recovery recommendations:         230 / 500
```

The detector is intentionally treated as imperfect.

False recovery recommendations introduce unnecessary computation and can expose originally sufficient questions to potential evidence loss.

Missed insufficient cases prevent the controller from attempting recovery when additional evidence may be needed.

These limitations must be considered when evaluating the complete recovery pipeline.

### Phase 10.5A — Controller contracts

Implementation:

```text
src/recovery/controller.py
```

The controller defines:

```text
ControllerDecision
    |
    +-- KEEP_ORIGINAL
    +-- ATTEMPT_RECOVERY
```

and:

```text
ControllerOutcome
    |
    +-- ORIGINAL_ACCEPTED
    +-- RECOVERY_PENDING_VERIFICATION
    +-- RECOVERY_FALLBACK
```

The controller receives:

```text
Original Query
Original Results
Runtime Diagnosis
```

It then decides whether to execute the configured recovery intervention.

A completed intervention with nonempty recovered results produces:

```text
RECOVERY_PENDING_VERIFICATION
```

It does not immediately replace the original evidence.

The controller falls back to the original evidence when:

- recovery execution fails,
- the recovery result is empty,
- the executor raises an exception,
- or the returned recovery metadata violates expected provenance.

Controller tests:

```powershell
python -m scripts.test_recovery_controller
```

### Phase 10.5B — Runtime integration

Implementation:

```text
src/recovery/runtime_integration.py
```

This module connects:

```text
Retrieved Results
      |
      v
Feature Extraction
      |
      v
Runtime Diagnosis
      |
      v
Recovery Controller
```

The integration constructs the same 29-feature representation used by the frozen detector.

It checks feature names and prevents accidental feature collisions.

The integrated result preserves:

```text
Feature Row
Runtime Diagnosis
Controller Result
```

This makes it possible to inspect the detector inputs and controller decisions separately.

Tests:

```powershell
python -m scripts.test_runtime_integration
```

### Phase 10.5C — Real feature-parity validation

A critical requirement is that runtime feature extraction must reproduce the feature representation used during detector development.

Otherwise, a detector that performs well on an offline feature table may behave differently during real retrieval.

The feature-parity experiment reran the retrieval pipeline on validation questions and compared runtime-generated features against the stored diagnostic features.

Results:

```text
Validation questions:        50

Features per question:       29

Feature comparisons:       1450

Matching comparisons:      1450

Mismatches:                   0
```

Therefore:

```text
Runtime feature parity: 1450 / 1450
```

This confirms exact feature-value agreement for the tested validation examples.

It does not establish that the detector's predictions are always correct.

Validation command:

```powershell
python -m scripts.test_real_feature_parity --max-questions 50
```

### Phase 10.5D — Real controller-routing validation

The next experiment executed the runtime diagnosis and controller-routing path on 50 validation questions.

The experiment used real retrieval results and the frozen detector.

A deterministic recovery probe was used to validate routing behavior.

**The experiment did not execute the full Qwen-based recovery intervention and therefore does not measure real end-to-end recovery quality.**

Results:

```text
Questions evaluated:              50

Correct routing decisions:        34

Routing accuracy:               68%

Recovery recommendations:         21

Original evidence retained:       29

False recovery recommendations:   10

Missed insufficient cases:         6
```

Confusion breakdown:

| Oracle evidence state | Keep original | Attempt recovery |
|---|---:|---:|
| Sufficient | 23 | 10 |
| Insufficient | 6 | 11 |

This means:

```text
Correct keep decisions:       23

Correct recovery decisions:   11

False recovery flags:         10

Missed recovery cases:         6
```

The experiment confirms that the frozen runtime detector can be connected to the recovery controller without relying on oracle labels for runtime routing.

It also demonstrates that the controller makes imperfect decisions.

The false recovery and missed recovery cases are important targets for subsequent end-to-end evaluation.

Validation command:

```powershell
python -m scripts.test_real_controller_routing --max-questions 50
```

---

## Phase 10.6 — Recovery Candidate Handoff

### Status: Complete

The final Phase 10 stage connects controller decisions to the evidence-selection module.

Implementation:

```text
src/recovery/recovery_handoff.py
```

The purpose is to prepare a recovery candidate for future verification without prematurely replacing the original evidence.

The handoff receives:

```text
ControllerResult
```

and produces:

```text
RecoveryHandoff
    |
    +-- original_results
    +-- selection
    +-- pending_verification
    +-- reason
```

When the controller keeps the original evidence or falls back after unsuccessful recovery, no verification candidate is created.

When recovery completes successfully and produces nonempty evidence, the handoff invokes the evidence selector.

The current selection policy uses the balanced 5/5 configuration.

The selected evidence is returned as a candidate with explicit provenance.

### Safety conditions

The handoff enforces several invariants:

1. The controller must not modify the original evidence before verification.
2. Pending recovery must contain a completed, nonempty recovery result.
3. The recovery result must refer to the same original evidence.
4. Evidence selection must produce a nonempty candidate.
5. Candidate preparation must not imply verified sufficiency.

Conceptually:

```text
Controller Result
       |
       v
Recovery Handoff
       |
       +--> No Recovery
       |         |
       |         v
       |    Original Evidence
       |
       +--> Recovery Candidate
                 |
                 v
          Evidence Selection
                 |
                 v
          Candidate Evidence
                 |
                 v
          Pending Verification
```

The original evidence remains available as the fallback.

### Validation

The recovery-handoff tests cover:

```text
No recovery produces no verification candidate

Balanced 5/5 selection retains provenance

Premature evidence replacement is rejected

Mismatched recovery provenance is rejected
```

All four tests passed.

```text
PHASE 10.6 RECOVERY HANDOFF TESTS PASSED
```

Validation command:

```powershell
python -m scripts.test_recovery_handoff
```

This establishes a controlled boundary between recovery execution and post-recovery verification.

---

## Phase 10 — Research Findings

The Phase 10 experiments support several observations.

### 1. Increasing retrieval depth does not guarantee complete evidence

Expanded candidate retrieval can expose additional evidence, but the final ranking stage may still exclude required supporting facts.

Candidate generation and final evidence selection must therefore be evaluated separately.

### 2. Multi-hop recovery depends on intermediate decisions

Evidence-conditioned second-hop retrieval requires successful entity extraction, bridge selection, and relation-query generation.

An error in any of these stages can prevent recovery from reaching the missing evidence.

### 3. Evidence-selection policy affects recovery quality

On the 50-question forced-recovery validation experiment, balanced 5/5 selection increased complete evidence coverage from:

```text
33 / 50
```

to:

```text
38 / 50
```

This is an improvement in annotated evidence completeness on the tested sample, not proof of improved answer accuracy.

### 4. Runtime recovery routing is imperfect

The frozen detector correctly routed 34 of 50 validation questions in the controller-routing experiment.

The remaining cases included both unnecessary recovery recommendations and missed insufficient-evidence cases.

This confirms that recovery quality must be evaluated together with detector quality.

### 5. Successful recovery execution is not verified evidence repair

A recovery intervention can complete successfully while returning irrelevant, redundant, or incomplete evidence.

The implementation therefore distinguishes:

```text
Recovery Executed
```

from:

```text
Recovery Candidate Prepared
```

and:

```text
Recovery Verified
```

The third stage remains future work.

### 6. Original evidence must remain available

Replacing the original evidence immediately after recovery creates the possibility of losing useful supporting facts.

Phase 10 preserves the original retrieval results and makes recovered evidence provisional until verification.

---

## Phase 10 — Current Limitations

### Runtime failure attribution

The current detector predicts evidence sufficiency rather than reliably identifying every fine-grained retrieval failure type.

The recovery controller therefore does not yet implement a fully validated failure-type-specific intervention policy.

### Query-generation reliability

The second-hop query-generation pipeline remains sensitive to bridge selection and generated-query validity.

In the broader 50-question experiment, 24 cases did not produce structurally valid second-hop retrieval queries.

### Limited recovery evaluation

The broad evidence-selection experiment evaluates 50 validation questions under forced recovery.

These results should not be interpreted as final held-out benchmark performance or as a measurement of the deployed controller's complete recovery behavior.

### Evidence-selection regressions

Balanced selection reserves slots for both retrieval hops but does not guarantee preservation of every original supporting fact.

A larger evaluation may reveal regressions not observed in the current sample.

### Detector errors

The frozen runtime detector produces false recovery recommendations and misses some insufficient-evidence cases.

These errors affect both computational cost and potential recovery benefit.

### Post-recovery verification

Phase 10 prepares recovery candidates but does not yet independently determine whether the recovered evidence is sufficient.

The current controller must not interpret:

```text
RECOVERY_PENDING_VERIFICATION
```

as successful evidence repair.

### Answer-level evaluation

The current experiments primarily measure retrieval behavior and HotpotQA annotated supporting-fact coverage.

They do not yet establish improvements in:

- final answer accuracy,
- citation correctness,
- answer faithfulness,
- hallucination reduction,
- abstention quality,
- or end-to-end latency and computational cost.

---

## Phase 10 — Repository Structure

The recovery implementation is organized under:

```text
src/recovery/
    |
    +-- __init__.py
    |
    +-- states.py
    +-- expanded_candidates.py
    +-- entity_extraction.py
    +-- bridge_selection.py
    +-- llm_bridge_selection.py
    +-- bridge_query.py
    +-- relation_query.py
    +-- qwen_generator.py
    +-- multi_query.py
    +-- second_hop.py
    |
    +-- evidence_selection.py
    |
    +-- controller.py
    +-- runtime_integration.py
    +-- recovery_handoff.py
```

The associated experiment documentation is stored in:

```text
docs/experiments/
    |
    +-- phase_10_4_evidence_selection.md
    +-- phase_10_4e_results.csv
```

The repository also contains separate scripts for component tests, recovery experiments, runtime feature-parity validation, and controller-routing evaluation.

---

## Phase 10 — Reproducibility

Commands should be executed from the repository root with the virtual environment activated.

### Recovery state and interventions

```powershell
python -m scripts.test_recovery_states
python -m scripts.test_expanded_candidate_recovery
python -m scripts.test_entity_extraction
python -m scripts.test_bridge_selection
python -m scripts.test_bridge_query
python -m scripts.test_relation_query
python -m scripts.test_second_hop
```

### Evidence selection

```powershell
python -m scripts.test_evidence_selection
python -m scripts.test_selection_policies
python -m scripts.evaluate_broad_evidence_selection
```

### Controller integration

```powershell
python -m scripts.test_recovery_controller
python -m scripts.test_runtime_integration
```

### Real feature-parity validation

```powershell
python -m scripts.test_real_feature_parity --max-questions 50
```

### Real controller-routing validation

```powershell
python -m scripts.test_real_controller_routing --max-questions 50
```

### Recovery handoff

```powershell
python -m scripts.test_recovery_handoff
```

Some experiments load Hugging Face models and perform CPU-based language-model inference.

Consequently, execution time and resource requirements vary substantially between unit tests and full recovery evaluations.

---

## Phase 10 — Conclusion

Phase 10 establishes the recovery foundation required to investigate adaptive evidence repair.

The project now supports:

```text
Initial Retrieval
       |
       v
Runtime Sufficiency Detection
       |
       v
Recovery Decision
       |
       v
Recovery Intervention
       |
       v
Evidence-Preserving Selection
       |
       v
Recovery Candidate Handoff
```

The experiments demonstrate that additional retrieval and evidence-selection strategies can improve annotated supporting-fact coverage on selected validation examples.

However, the current results do not yet establish that a complete detector-triggered recovery system consistently improves evidence sufficiency or final answer quality.

The next stage is:

**Phase 11 — Post-Recovery Evidence Verification**

Phase 11 will investigate whether a recovery candidate can be accepted, rejected, or marked uncertain using inference-time evidence signals without accessing benchmark gold labels.

It will also evaluate recovery repairs and regressions against HotpotQA supporting facts offline.

The central research objective remains:

> **Detect retrieval insufficiency, recover missing evidence when appropriate, and verify whether the intervention actually improved the evidence before proceeding to answer generation.**

---

# Author

**Sahithi Vankayala**

GitHub: Sahithi102302