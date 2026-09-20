# AdaptiveRAG

## Failure-Aware Retrieval and Self-Correcting Generation

AdaptiveRAG is a research-oriented Retrieval-Augmented Generation (RAG) project investigating whether a RAG system can identify **why retrieval failed** and dynamically select an appropriate recovery strategy instead of applying the same retrieval pipeline to every query.

The central research question is:

> **Can a RAG system identify the specific reason retrieval failed and dynamically select an appropriate recovery strategy, improving answer quality and faithfulness without excessive latency or cost?**

This repository is being developed incrementally. Each major component is implemented, inspected, and validated before the next stage is added.

> **Current status:** The data ingestion, baseline chunking, HotpotQA corpus construction, gold-evidence mapping, and corpus-integrity validation stages are complete. Adaptive failure diagnosis and recovery are planned but are not yet implemented.

---

## Table of Contents

* [Motivation](#motivation)
* [Research Question](#research-question)
* [Core Idea](#core-idea)
* [Failure Taxonomy](#failure-taxonomy)
* [Planned Recovery Strategies](#planned-recovery-strategies)
* [Research Hypotheses](#research-hypotheses)
* [Baseline Progression](#baseline-progression)
* [Dataset Strategy](#dataset-strategy)
* [Why HotpotQA](#why-hotpotqa)
* [HotpotQA Corpus Engineering](#hotpotqa-corpus-engineering)
* [Dataset Integrity Findings](#dataset-integrity-findings)
* [Current Implementation](#current-implementation)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Running the Current Pipeline](#running-the-current-pipeline)
* [Planned Evaluation](#planned-evaluation)
* [Roadmap](#roadmap)
* [Research Engineering Principles](#research-engineering-principles)
* [Author](#author)

---

## Motivation

A conventional RAG pipeline usually follows a static workflow:

```text
User Query
    |
    v
Retrieve Top-K Documents
    |
    v
Construct Context
    |
    v
Generate Answer
```

This architecture assumes that retrieval has provided useful evidence.

In practice, retrieval can fail in several fundamentally different ways.

For example:

* the retriever may select the wrong document,
* the correct document may be retrieved but the wrong passage selected,
* some required evidence may be missing,
* the available information may be outdated,
* multiple sources may conflict,
* the query itself may be ambiguous,
* or the language model may make an inference that is unsupported by the retrieved evidence.

These failures should not necessarily trigger the same response.

A system suffering from an ambiguous query may need clarification.

A system missing one part of a multi-hop question may need additional retrieval.

A system retrieving the correct document but the wrong passage may need context expansion.

A system facing outdated information may need temporal filtering.

A system with insufficient evidence may need to abstain rather than generate an unsupported answer.

AdaptiveRAG investigates whether retrieval failures can be explicitly diagnosed and routed to targeted recovery strategies.

---

## Research Question

The primary research question is:

> **Can a RAG system identify the specific reason retrieval failed and dynamically select an appropriate recovery strategy, improving answer quality and faithfulness without excessive latency or cost?**

The project separates three related but different problems:

```text
Detection
    |
    | Something is wrong.
    v
Diagnosis
    |
    | What specifically went wrong?
    v
Recovery
    |
    | What action should be taken?
    v
Verification
```

This distinction is central to the project.

A system may successfully detect weak evidence without knowing why the evidence is weak.

Similarly, correctly diagnosing a failure does not automatically imply that the selected recovery action will solve it.

AdaptiveRAG therefore studies these stages separately.

---

## Core Idea

A key principle behind the project is:

> **Semantic similarity is not the same as evidence sufficiency.**

A retrieved passage can be semantically similar to a query while still failing to contain the evidence necessary to answer it.

The planned AdaptiveRAG architecture is:

```text
User Query
    |
    v
Query Processing
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
    +-------------------- Yes --------------------+
    |                                             |
    No                                            v
    |                                      Generate Answer
    v
Failure Diagnosis
    |
    v
Recovery Router
    |
    v
Select Recovery Strategy
    |
    v
Re-retrieval
    |
    v
Evidence Verification
    |
    +-------------------- Sufficient ------------> Generate Answer
    |
    +-------------------- Insufficient
                              |
                              v
                     Recovery budget remaining?
                              |
                       +------+------+
                       |             |
                      Yes            No
                       |             |
                       v             v
                 Recover Again     Abstain
```

The goal is not simply to add more retrieval components.

The research question is whether the system can decide **when** recovery is necessary and **which recovery action** is appropriate.

---

## Failure Taxonomy

AdaptiveRAG is designed to study several categories of retrieval and generation failure.

### 1. Wrong Document

The retrieved documents are related to the query but do not contain the required evidence.

Potential recovery:

* query rewriting,
* hybrid retrieval,
* reranking,
* metadata filtering.

### 2. Correct Document, Wrong Chunk

The correct source document is present, but the retrieved chunk does not contain the necessary evidence.

Potential recovery:

* neighboring-chunk expansion,
* sentence-aware retrieval,
* reranking,
* larger context windows.

### 3. Insufficient Context

Some relevant evidence is retrieved, but the context is incomplete.

This is especially important for multi-hop questions.

Potential recovery:

* query decomposition,
* multi-hop retrieval,
* additional retrieval,
* context expansion.

### 4. Outdated Information

Retrieved evidence may have been correct historically but is no longer current.

Potential recovery:

* temporal filtering,
* freshness-aware retrieval,
* newer-source retrieval.

### 5. Conflicting Sources

Multiple retrieved sources provide incompatible claims.

Potential recovery:

* source verification,
* evidence comparison,
* conflict-resolution logic,
* abstention when the conflict cannot be resolved reliably.

### 6. Ambiguous Query

The query has multiple plausible interpretations.

Potential recovery:

* query clarification,
* interpretation detection,
* targeted follow-up questions.

### 7. Unsupported Generation

Relevant context may exist, but the generated answer contains claims that are not supported by the evidence.

Potential recovery:

* evidence verification,
* answer regeneration,
* citation checking,
* abstention.

---

## Planned Recovery Strategies

The planned recovery-action space includes:

* query rewriting,
* query decomposition,
* hybrid retrieval,
* reranking,
* multi-hop retrieval,
* neighboring-context expansion,
* metadata filtering,
* temporal filtering,
* source verification,
* conflict resolution,
* clarification,
* and abstention.

The recovery mechanism can eventually be formulated as a routing policy:

```text
state -> recovery action
```

where the state may contain signals such as:

* retrieval scores,
* dense/sparse retrieval agreement,
* evidence coverage,
* retrieved-document metadata,
* temporal information,
* contradiction signals,
* entailment signals,
* and evidence-sufficiency judgments.

---

## Research Hypotheses

The project is designed around four primary hypotheses.

### H1 - Failure-Aware Retrieval

Failure-aware retrieval will improve evidence recall on difficult queries compared with static retrieval pipelines.

### H2 - Failure-Specific Recovery

Different retrieval failure types will benefit from different recovery strategies.

### H3 - Quality vs. System Cost

Adaptive recovery can improve answer faithfulness without proportional increases in latency and computational or API cost.

### H4 - Evidence Sufficiency

Explicit evidence-sufficiency detection can reduce unnecessary recovery operations.

These are **research hypotheses**, not experimental results.

No performance claim should be inferred from them until the corresponding experiments have been implemented and evaluated.

---

## Baseline Progression

AdaptiveRAG will be evaluated against increasingly capable static baselines.

### Baseline 0 - LLM Only

```text
Query
  |
  v
LLM
  |
  v
Answer
```

No retrieval.

This provides a reference point for measuring the value of external evidence.

### Baseline 1 - Dense Vector RAG

```text
Query
  |
  v
Embedding Model
  |
  v
Vector Search
  |
  v
Top-K Context
  |
  v
LLM
```

### Baseline 2 - Hybrid RAG

```text
             +-> Dense Retrieval --+
Query -------|                      +-> Fusion -> Context
             +-> Sparse Retrieval -+
```

Dense semantic retrieval will be combined with sparse lexical retrieval such as BM25.

### Baseline 3 - Hybrid Retrieval + Reranking

```text
Dense Retrieval
       +
Sparse Retrieval
       |
       v
Candidate Fusion
       |
       v
Cross-Encoder Reranker
       |
       v
Top Evidence
```

### AdaptiveRAG

```text
Retrieval
    |
    v
Evidence Inspection
    |
    v
Failure Diagnosis
    |
    v
Adaptive Recovery
    |
    v
Verification
    |
    +----> Generate
    |
    +----> Recover Again
    |
    +----> Abstain
```

This progression is important because it helps distinguish improvements caused by stronger retrieval from improvements caused specifically by adaptive recovery.

---

## Dataset Strategy

AdaptiveRAG uses different datasets for different experimental purposes rather than merging every benchmark into one large dataset.

### HotpotQA

Primary development and evidence-oriented benchmark.

Used for studying:

* multi-hop retrieval,
* missing evidence,
* wrong-document retrieval,
* wrong-passage retrieval,
* evidence sufficiency,
* evidence coverage,
* query decomposition,
* context expansion,
* and supporting-fact retrieval.

### BEIR - Planned

Planned for retrieval benchmarking.

Potential uses include comparing:

* dense retrieval,
* BM25,
* hybrid retrieval,
* reranking,
* Recall@K,
* Precision@K,
* MRR,
* and nDCG.

### CRAG - Planned

Planned for reliability-oriented experiments involving changing or time-sensitive information.

Potential use cases include:

* outdated evidence,
* changing facts,
* retrieval reliability,
* and abstention.

### RAGBench - Planned

Planned as an additional RAG evaluation resource for broader generalization experiments.

### AdaptiveRAG-FailureBench - Planned

A controlled evaluation benchmark is planned specifically for AdaptiveRAG.

Its purpose will be to evaluate the complete chain:

```text
Known Failure
     |
     v
Failure Diagnosis
     |
     v
Recovery Selection
     |
     v
Recovery Execution
     |
     v
Recovery Success
```

This is important because aggregate RAG accuracy alone cannot determine whether a failure-aware recovery mechanism is behaving correctly.

---

## Why HotpotQA

HotpotQA provides question-answer pairs together with supporting-fact annotations.

A simplified example looks like:

```text
Question
    |
    +-> Context Document A
    |       |
    |       +-> Sentence 0
    |       +-> Sentence 1
    |       +-> Sentence 2
    |
    +-> Context Document B
            |
            +-> Sentence 0
            +-> Sentence 1

Gold Evidence:
(Document A, Sentence 1)
(Document B, Sentence 0)
```

This makes it useful for evaluating not only whether the final answer is correct, but whether the retrieval system actually recovered the evidence required to construct that answer.

The project currently uses the HotpotQA `distractor` configuration.

The training split contains:

```text
90,447 QA examples
```

and the validation split contains:

```text
7,405 QA examples
```

An important limitation is that the distractor configuration provides a small candidate context for each question.

Retrieval among only those candidate documents should **not** be treated as equivalent to open-corpus retrieval.

AdaptiveRAG therefore uses the annotations for evidence analysis while constructing a larger deduplicated retrieval corpus across examples for retrieval experiments.

---

## HotpotQA Corpus Engineering

Before building embeddings or retrieval indexes, the project performs explicit corpus-integrity analysis.

This revealed several preprocessing issues that could otherwise silently affect retrieval evaluation.

### Raw Training Corpus

The full HotpotQA training split contains:

```text
QA examples:                 90,447
Document appearances:      899,667
Unique normalized titles:  482,021
```

A document appearance represents a document occurring inside a question's context.

The same source can therefore appear multiple times across different questions.

---

### Why Title-Only Deduplication Was Rejected

A natural first approach would be to use the normalized Wikipedia title as the document identifier.

However, a full corpus audit found:

```text
Titles with multiple content versions: 1,661
```

Therefore:

```text
title -> document
```

is not always a safe one-to-one mapping.

Two documents can share the same normalized title while containing different text.

AdaptiveRAG therefore defines document identity using:

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

The current implementation uses the first 16 hexadecimal characters of the SHA-256 digest.

Document IDs therefore follow the general structure:

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

which exactly matches the number of normalized titles found to have multiple content versions in the audit.

This agreement provides an independent consistency check on the corpus-construction logic.

---

## Sentence-Level Evidence Preservation

HotpotQA represents supporting evidence using document titles and sentence positions.

Conceptually:

```text
(title, sentence_id)
```

Because sentence IDs are positional, preprocessing must preserve sentence structure.

For example:

```text
Sentence 0
Sentence 1
Sentence 2
Sentence 3
```

cannot safely become:

```text
Sentence 0
Sentence 1
Sentence 3
```

simply because Sentence 2 happened to be empty.

Doing so would shift subsequent sentence indices and corrupt the mapping between the source benchmark and the processed corpus.

The HotpotQA adapter therefore preserves sentence positions, including empty sentence positions.

Flattened document text can omit empty strings for readability, while the original positional sentence structure remains available in metadata.

---

## Gold Evidence Representation

Internally, a supporting fact is represented using:

```text
document_id
title
sentence_id
```

The `document_id` identifies the exact content version of the source document.

This is important because title alone is insufficient when multiple content versions exist.

Each question's gold supporting facts are resolved against the exact document versions appearing in that question's context.

This creates the foundation for later evidence-aware retrieval metrics.

---

## Dataset Integrity Findings

The full HotpotQA training corpus was independently audited before the finalized adapter was accepted.

The audit found:

```text
Total supporting facts:           215,684
Valid supporting facts:           215,662
Missing supporting documents:           0
Invalid sentence IDs:                   22
Gold facts on empty sentences:           0
```

There are therefore:

```text
22
```

supporting-fact annotations whose sentence IDs do not correspond to valid sentence positions in the associated source document.

These anomalies originate from the source benchmark annotations.

AdaptiveRAG does **not silently repair these labels**.

Instead, malformed annotations are:

1. preserved,
2. detected,
3. reported,
4. and intended to be handled explicitly by the evaluation policy.

This follows an important research-engineering principle:

> **Preserve source annotations, validate them, and document anomalies instead of silently modifying benchmark ground truth.**

---

## Independent Corpus Validation

Two separate stages were used to validate the corpus.

### Raw Corpus Audit

The raw dataset was inspected independently of the final adapter.

It identified:

```text
90,447 QA examples
899,667 document appearances
482,021 unique normalized titles
1,661 titles with multiple content versions
538 documents containing empty sentence positions
538 total empty sentence positions
215,684 supporting facts
215,662 valid supporting facts
22 invalid sentence IDs
0 missing supporting documents
0 gold facts pointing to empty sentences
```

### Final Adapter Validation

The finalized adapter independently produced:

```text
90,447 QA examples
483,682 unique document versions
215,684 supporting facts
215,662 valid supporting facts
0 missing supporting documents
22 invalid sentence IDs
0 gold facts pointing to empty sentences
```

The agreement between the independent raw audit and finalized adapter provides a useful integrity check before retrieval experiments begin.

---

## Current Implementation

The project is being implemented incrementally.

### Phase 1 - Generic Document Ingestion

**Status: Complete**

Implemented:

* reusable `Document` abstraction,
* TXT loading,
* JSON loading,
* directory loading,
* metadata propagation,
* input validation,
* empty-document handling,
* document statistics.

The generic ingestion layer is intentionally independent of HotpotQA so future datasets and document sources can reuse the same internal representation.

---

### Phase 2 - Character Chunking Baseline

**Status: Complete**

Implemented:

* reusable `Chunk` abstraction,
* configurable character chunk size,
* configurable chunk overlap,
* stable chunk IDs,
* source-document metadata propagation,
* chunk statistics.

The character chunker is retained as a baseline.

It is not assumed to be the optimal chunking strategy.

---

### Phase 3 - HotpotQA Inspection

**Status: Complete**

Completed:

* Hugging Face dataset integration,
* schema inspection,
* question inspection,
* context-document inspection,
* supporting-fact inspection,
* train/validation split inspection.

Observed dataset schema:

```text
id
question
answer
type
level
supporting_facts
context
```

The inspection stage established how questions, context documents, sentences, and gold supporting facts are related before corpus transformation was implemented.

---

### Phase 4 - HotpotQA Corpus Adapter

**Status: Complete**

Implemented:

* dedicated HotpotQA adapter,
* sentence-position preservation,
* normalized-title handling,
* version-aware document identity,
* global document deduplication,
* exact supporting-fact mapping,
* evidence validation,
* malformed-evidence detection,
* full-corpus integrity audit.

Final training-corpus summary:

```text
QA examples:               90,447
Unique document versions: 483,682
Gold supporting facts:    215,684
Valid supporting facts:   215,662
Malformed sentence IDs:        22
```

---

### Phase 5 - Evidence-Aware Chunking

**Status: Planned / Next**

The next stage will introduce sentence-aware chunking.

Instead of representing chunks only using character ranges, evidence-aware chunks will preserve sentence mappings such as:

```text
Chunk
|
+-- document_id
+-- title
+-- chunk_index
+-- sentence_start
+-- sentence_end
+-- sentence_ids
+-- text
```

This will allow a gold supporting fact:

```text
(document_id, sentence_id)
```

to be mapped deterministically to one or more chunks containing that evidence.

The existing character chunker will remain available as a baseline so chunking strategies can be compared experimentally rather than assuming sentence-aware chunking is automatically superior.

---

## Repository Structure

Current repository structure:

```text
adaptive-rag/
|
+-- data/
|   |
|   +-- demo/
|       +-- company_history.txt
|       +-- company_updates.json
|
+-- scripts/
|   +-- audit_hotpotqa_corpus.py
|   +-- inspect_hotpotqa.py
|   +-- inspect_hotpotqa_duplicate.py
|   +-- test_chunking.py
|   +-- test_hotpotqa_adapter.py
|   +-- test_ingestion.py
|   +-- validate_hotpotqa_full.py
|
+-- src/
|   |
|   +-- chunking/
|   |   +-- __init__.py
|   |   +-- chunk.py
|   |   +-- chunker.py
|   |
|   +-- ingestion/
|       +-- __init__.py
|       +-- document.py
|       +-- hotpotqa.py
|       +-- loader.py
|
+-- .gitignore
+-- README.md
+-- requirements.txt
```

The repository structure will expand as retrieval, reranking, generation, diagnosis, recovery, and evaluation components are implemented.

---

## Installation

### Requirements

Current development environment:

```text
Python 3.11
VS Code
Windows PowerShell
```

The project has been developed and validated using Python 3.11.

### Clone the Repository

Once published:

```bash
git clone https://github.com/Sahithi102302/adaptive-rag.git
cd adaptive-rag
```

### Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

Current core dependencies include:

```text
datasets==5.0.1
pyarrow==21.0.0
pandas==3.0.6
numpy==2.4.6
```

`pyarrow==21.0.0` is currently pinned because it has been validated with the project's current Hugging Face `datasets` setup.

Additional dependencies will be introduced incrementally as retrieval and generation components are implemented.

---

## Running the Current Pipeline

Commands should be run from the repository root.

### Test Generic Ingestion

```powershell
python -m scripts.test_ingestion
```

This validates the generic document ingestion pipeline using the small demo fixtures under `data/demo/`.

---

### Test Character Chunking

```powershell
python -m scripts.test_chunking
```

This validates the current character-based chunking implementation.

---

### Inspect HotpotQA

```powershell
python -m scripts.inspect_hotpotqa
```

This downloads/loads HotpotQA through Hugging Face `datasets` and inspects its schema and example structure.

---

### Test the HotpotQA Adapter

```powershell
python -m scripts.test_hotpotqa_adapter
```

This runs the adapter against a smaller subset before full-corpus validation.

A previously validated 1,000-example run produced:

```text
QA examples:                 1,000
Unique document versions:    9,758
Total document characters:   5,262,236
Total document words:        863,511
Average words/document:      88.49
Total supporting facts:      2,377
Valid supporting facts:      2,376
Missing documents:           0
Invalid sentence IDs:        1
Empty gold sentences:        0
```

---

### Audit the Raw HotpotQA Corpus

```powershell
python -m scripts.audit_hotpotqa_corpus
```

This independently examines corpus characteristics and annotation anomalies before relying on the transformed representation.

---

### Validate the Full Adapter

```powershell
python -m scripts.validate_hotpotqa_full
```

This validates the complete HotpotQA training corpus after version-aware corpus construction.

The validated run produced:

```text
QA examples:                 90,447
Unique document versions:    483,682

Total supporting facts:      215,684
Valid supporting facts:      215,662
Missing documents:           0
Invalid sentence IDs:        22
Empty gold sentences:        0

Total detected problems:     22
```

---

## Planned Evaluation

AdaptiveRAG will be evaluated at several different levels.

### Retrieval Metrics

Planned metrics include:

* Recall@K
* Precision@K
* Hit Rate@K
* Mean Reciprocal Rank (MRR)
* nDCG
* document recall
* supporting-fact recall
* complete-evidence recall

Retrieval evaluation will distinguish between retrieving a relevant document and retrieving the exact evidence required to answer a question.

---

### Generation Metrics

Planned metrics include:

* answer correctness,
* answer relevance,
* faithfulness,
* citation correctness.

---

### Adaptation Metrics

The adaptive layer requires metrics beyond conventional RAG evaluation.

Planned metrics include:

* failure-diagnosis accuracy,
* recovery success rate,
* unnecessary recovery rate.

For example, a system that always triggers expensive recovery might improve some retrieval metrics while still being a poor adaptive system.

---

### System Metrics

Planned operational metrics include:

* end-to-end latency,
* retrieval latency,
* reranking latency,
* token usage,
* model/API calls,
* computational cost.

These measurements are necessary because adaptive recovery introduces additional computation.

---

### Reliability Metrics

Planned reliability analysis includes:

* hallucination rate,
* unsupported-answer rate,
* abstention behavior.

The goal is not only to maximize answer rate.

A reliable system should be capable of recognizing situations in which the available evidence does not justify an answer.

---

## Evaluation by Failure Type

Aggregate accuracy can hide important behavior.

AdaptiveRAG therefore plans to report performance by failure category.

Conceptually:

```text
                     Baseline    Adaptive
Wrong document          ?           ?
Wrong chunk             ?           ?
Missing evidence        ?           ?
Outdated evidence       ?           ?
Conflicting evidence    ?           ?
Ambiguous query         ?           ?
Generation failure      ?           ?
```

The question marks are intentional.

No values will be added until the corresponding experiments have been run.

This breakdown is necessary to determine whether particular recovery strategies actually solve the failure types they were designed to address.

---

## Roadmap

### Completed

* [x] Project research framing
* [x] Generic document representation
* [x] TXT and JSON ingestion
* [x] Character chunking baseline
* [x] HotpotQA integration
* [x] HotpotQA schema inspection
* [x] Full raw-corpus audit
* [x] Version-aware document identity
* [x] Corpus deduplication
* [x] Gold supporting-fact mapping
* [x] Gold-evidence validation
* [x] Full adapter validation

### Next

* [ ] Sentence-aware chunk representation
* [ ] Evidence-aware chunking
* [ ] Gold evidence-to-chunk mapping
* [ ] Character vs. sentence chunking comparison

### Retrieval

* [ ] Embedding pipeline
* [ ] Dense vector index
* [ ] Dense retrieval baseline
* [ ] BM25 sparse retrieval
* [ ] Hybrid retrieval
* [ ] Rank fusion
* [ ] Cross-encoder reranking

### Generation

* [ ] Context construction
* [ ] Baseline answer generation
* [ ] Citation-aware generation
* [ ] Faithfulness evaluation

### Failure Awareness

* [ ] Evidence-sufficiency detection
* [ ] Retrieval failure signals
* [ ] Failure taxonomy implementation
* [ ] Rule-based failure diagnosis
* [ ] LLM-assisted diagnosis
* [ ] Learned diagnosis experiments

### Adaptive Recovery

* [ ] Recovery router
* [ ] Query rewriting
* [ ] Query decomposition
* [ ] Context expansion
* [ ] Multi-hop recovery
* [ ] Metadata-aware recovery
* [ ] Temporal recovery
* [ ] Verification
* [ ] Recovery budget
* [ ] Abstention

### Evaluation

* [ ] Retrieval evaluation harness
* [ ] Generation evaluation harness
* [ ] Diagnosis evaluation
* [ ] Recovery evaluation
* [ ] Failure-type evaluation
* [ ] Latency measurement
* [ ] Cost measurement
* [ ] Reliability analysis
* [ ] Baseline comparisons

### Additional Benchmarks

* [ ] BEIR evaluation
* [ ] CRAG evaluation
* [ ] RAGBench evaluation
* [ ] AdaptiveRAG-FailureBench

---

## Research Engineering Principles

This project follows several engineering principles intended to keep experimental conclusions trustworthy.

### 1. Validate Before Scaling

Components are first tested on small subsets before full-corpus execution.

### 2. Preserve Ground Truth

Source annotations are not silently modified simply because they appear malformed.

### 3. Separate Retrieval Relevance from Evidence Sufficiency

A semantically relevant result is not automatically sufficient evidence.

### 4. Compare Against Strong Baselines

Adaptive behavior should be compared against progressively stronger static retrieval systems.

### 5. Measure Cost as Well as Quality

A recovery strategy is not automatically useful if its quality improvement requires disproportionate latency or computational cost.

### 6. Evaluate Failures Separately

Aggregate performance alone cannot explain whether the system actually diagnoses and repairs retrieval failures.

### 7. Keep Implemented and Planned Work Distinct

Repository documentation distinguishes between:

* implemented components,
* validated results,
* planned experiments,
* and research hypotheses.

Future functionality is not presented as completed work.

---

## Current Project Status

AdaptiveRAG is under active development.

The project has currently completed the corpus-engineering foundation required for retrieval experiments.

The immediate next stage is:

> **Phase 5 - Evidence-Aware Sentence Chunking**

After evidence-aware chunking is validated, development will proceed toward embeddings and the first real dense-retrieval baseline.

No claims about adaptive retrieval performance, answer-quality improvement, latency improvement, or hallucination reduction are made at the current stage because those experiments have not yet been implemented.

---

## Author

**Sahithi Vankayala**

GitHub: Sahithi102302
