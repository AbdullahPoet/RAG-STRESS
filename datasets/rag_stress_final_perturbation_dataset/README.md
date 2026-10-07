# RAG-Stress Dataset

Dataset package for **RAG-Stress**, a controlled benchmark for studying failure modes in Retrieval-Augmented Generation (RAG).

The benchmark is derived from a 500-question HotpotQA subset with `yes` / `no` answers excluded. It evaluates model robustness under three controlled perturbations:

1. **Topical Distractor Injection**
2. **Evidence Position Bias**
3. **Counterfactual Evidence Injection**

The package also contains a clean baseline, hard-negative audit data, skipped-counterfactual logs, and quality reports.

---

## Dataset Overview

| File | Rows | Purpose |
|---|---:|---|
| `clean_500.jsonl` | 500 | Clean baseline using only gold supporting context |
| `distractor_D1_D3_D5_500.jsonl` | 1,500 | Topical hard-negative experiment with 1, 3, or 5 distractors |
| `position_begin_middle_end_500.jsonl` | 1,500 | Position-bias experiment with gold evidence at beginning, middle, or end |
| `counterfactual_high_confidence_fixed.jsonl` | 431 | Final validated counterfactual experiment dataset |
| `counterfactual_skipped_fixed.jsonl` | 69 | Audit log of examples excluded from counterfactual generation |
| `hard_negative_audit_500.jsonl` | 500 | Per-question audit of the five selected hard negatives |
| `quality_report.json` | 1 report | Distractor/position generation statistics |
| `counterfactual_quality_report.json` | 1 report | Final counterfactual validation statistics |

> **Important:** `counterfactual_high_confidence_fixed.jsonl` is the counterfactual dataset that should be used for experiments.  
> `counterfactual_high_confidence.jsonl` is an older version and is superseded by the fixed file.

---

# 1. Clean Baseline

**File:** `clean_500.jsonl`

Contains one clean example for each of the 500 source questions.

Only the gold supporting context is included.

### Example schema

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "experiment": "clean",
  "condition": "clean",
  "context_chunks": [
    {
      "title": "...",
      "text": "...",
      "role": "gold"
    }
  ],
  "context": "..."
}
```

### Purpose

The clean condition establishes baseline QA performance:

\[
A_{clean}
\]

All perturbation results should be interpreted relative to this baseline.

---

# 2. Topical Distractor Experiment

**File:** `distractor_D1_D3_D5_500.jsonl`

Contains three perturbation levels for every source question:

| Condition | Distractors | Rows |
|---|---:|---:|
| `D1` | 1 | 500 |
| `D3` | 3 | 500 |
| `D5` | 5 | 500 |

Total:

\[
500 \times 3 = 1500
\]

examples.

### Experimental design

The gold context is preserved while semantically related but non-answer passages are added.

```text
D1:
Gold + 1 hard distractor

D3:
Gold + 3 hard distractors

D5:
Gold + 5 hard distractors
```

The construction is nested:

```text
D1 ⊂ D3 ⊂ D5
```

The strongest distractor used in `D1` remains present in `D3` and `D5`.

This ensures that the experimental variable is primarily **distractor density**, rather than an unrelated change in retrieved content.

### Hard-negative constraints

Every selected distractor:

- excludes the normalized gold answer
- does not serve as gold supporting evidence
- satisfies a minimum semantic score of `0.35`
- is ranked for similarity to both the question and the gold context

The hard-negative score used during construction was:

\[
S(d)=0.45\,sim(q,d)+0.55\,sim(g,d)
\]

where:

- \(q\) = question
- \(g\) = gold context
- \(d\) = candidate distractor

### Dataset-level hard-negative statistics

There are 2,500 selected hard negatives across the 500 source questions.

| Statistic | Score |
|---|---:|
| Minimum | 0.3504 |
| Mean | 0.7017 |
| Median | 0.7103 |
| P10 | 0.4722 |
| P25 | 0.5881 |
| P75 | 0.8333 |
| P90 | 0.9102 |

### Hard-negative sources

| Source | Count |
|---|---:|
| Question-specific HotpotQA distractors | 1,340 |
| Non-supporting text from gold documents | 1,034 |
| Global HotpotQA fallback distractors | 109 |
| Gold answer-masked evidence | 13 |
| Combined answer-masked evidence | 4 |

### Example schema

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "experiment": "distractor",
  "condition": "D3",
  "num_distractors": 3,
  "minimum_semantic_score": 0.35,
  "gold_indices": [0, 1],
  "context_chunks": [
    {
      "title": "...",
      "text": "...",
      "role": "gold"
    },
    {
      "title": "...",
      "text": "...",
      "role": "distractor",
      "semantic_score": 0.72,
      "question_similarity": 0.68,
      "gold_similarity": 0.75
    }
  ],
  "context": "..."
}
```

### Main research question

> How does answer quality change as the number of semantically relevant but non-answer passages increases?

Recommended metrics:

- Exact Match (EM)
- Token F1
- Accuracy degradation relative to clean

\[
\Delta A_n=A_{clean}-A_{D_n}
\]

---

# 3. Evidence Position Bias Experiment

**File:** `position_begin_middle_end_500.jsonl`

Contains three variants for each of the 500 source questions:

| Condition | Rows |
|---|---:|
| `beginning` | 500 |
| `middle` | 500 |
| `end` | 500 |

Total:

\[
500 \times 3 = 1500
\]

examples.

### Experimental control

Each question uses:

- the same question
- the same gold passages
- the same four hard distractors
- the same number of chunks

Only the **location of the gold evidence block** changes.

```text
Beginning:
[GOLD]
[GOLD]
[D1]
[D2]
[D3]
[D4]

Middle:
[D1]
[D2]
[GOLD]
[GOLD]
[D3]
[D4]

End:
[D1]
[D2]
[D3]
[D4]
[GOLD]
[GOLD]
```

Because HotpotQA can require multi-hop evidence, the complete gold evidence block is moved together.

### Example schema

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "experiment": "position",
  "condition": "middle",
  "num_distractors": 4,
  "gold_indices": [2, 3],
  "context_chunks": [...],
  "context": "..."
}
```

### Main research question

> Does the model use the same relevant evidence equally well when it appears at different positions in the retrieved context?

Recommended metrics:

- Exact Match
- Token F1
- performance by `beginning`, `middle`, and `end`

---

# 4. Counterfactual Experiment

**Primary file:** `counterfactual_high_confidence_fixed.jsonl`

Contains **431 validated counterfactual examples**.

The remaining 69 source questions were deliberately excluded when a reliable automatic counterfactual could not be produced.

### Construction

A gold fact is modified while preserving the surrounding supporting context as much as possible.

Example:

```text
Original:
The company was founded in 1998.

Counterfactual:
The company was founded in 2005.
```

or:

```text
Original:
The headquarters are in Delhi.

Counterfactual:
The headquarters are in Toronto.
```

The fixed generator modifies answer-bearing content in both:

- passage text
- document titles

This prevents the original answer from leaking through metadata.

### Replacement priorities

Counterfactual replacements were generated using:

1. a natural alternative already present in the question, when available
2. a type-consistent replacement bank
3. deterministic numeric transformation for numeric/date answers

Replacement-source counts:

| Source | Count |
|---|---:|
| Type-consistent bank | 231 |
| Numeric transform | 85 |
| Natural question alternative | 115 |

### Counterfactual type distribution

| Type | Count |
|---|---:|
| ENTITY | 246 |
| NUMERIC | 85 |
| PERSON | 44 |
| CITY | 15 |
| COUNTRY | 11 |
| LOCATION | 11 |
| FILM | 5 |
| COMPANY | 4 |
| BAND | 3 |
| TV_SHOW | 2 |
| ORGANIZATION | 2 |
| BOOK | 1 |
| GENRE | 1 |
| MATERIAL | 1 |

### Strict validation guarantees

For the final fixed file:

```text
Gold == counterfactual:       0
Counterfactual missing:       0
Residual gold leakage:        0
Rows with no applied change:  0
```

### Example schema

```json
{
  "id": "...",
  "question": "...",
  "experiment": "counterfactual",
  "condition": "counterfactual",
  "gold_answer": "Delhi",
  "counterfactual_answer": "Toronto",
  "counterfactual_type": "CITY",
  "replacement_source": "type_consistent_bank",
  "changed_passages": [...],
  "context_chunks": [...],
  "context": "..."
}
```

### Evaluation taxonomy

A model prediction can be classified as:

```text
GOLD_FOLLOW
COUNTERFACTUAL_FOLLOW
OTHER
```

A useful metric is Counterfactual Follow Rate:

\[
CFR =
\frac{\text{predictions matching the injected counterfactual}}
{\text{counterfactual examples}}
\]

This measures how strongly the model follows retrieved evidence when it conflicts with the original dataset fact.

---

# 5. Counterfactual Skip Log

**File:** `counterfactual_skipped_fixed.jsonl`

This file is **not an evaluation dataset**.

It records the 69 questions excluded from automatic counterfactual generation.

Skip reasons:

| Reason | Count |
|---|---:|
| Generic text answer | 49 |
| Answer not verbatim in supporting sentence | 8 |
| Residual gold answer after replacement | 12 |

Example:

```json
{
  "id": "...",
  "reason": "generic_text_answer"
}
```

This file exists for reproducibility and transparency.

Do not send it to the model during evaluation.

---

# 6. Hard-Negative Audit

**File:** `hard_negative_audit_500.jsonl`

Contains the five selected hard negatives for each source question.

Typical structure:

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "hard_negatives": [
    {
      "title": "...",
      "text": "...",
      "source": "local_hotpotqa",
      "question_similarity": 0.71,
      "gold_similarity": 0.80,
      "semantic_score": 0.76
    }
  ]
}
```

Use this file to:

- manually inspect hard-negative quality
- analyze similarity distributions
- identify the source of each selected distractor
- reproduce D1/D3/D5 construction

It is an **audit artifact**, not a direct LLM evaluation file.

---

# 7. Quality Reports

## `quality_report.json`

Contains generation statistics for the clean, distractor, and position datasets, including:

- source question count
- number of generated rows
- hard-negative count
- semantic-score threshold
- similarity distribution statistics
- hard-negative source distribution

## `counterfactual_quality_report.json`

Contains final counterfactual validation statistics, including:

- 500 source questions
- 431 accepted counterfactual examples
- 69 skipped examples
- zero residual gold leakage
- counterfactual type counts
- replacement-source counts
- skip reasons

---

# 8. Recommended Dataset Layout

```text
datasets/
│
├── clean_500.jsonl
│
├── distractor_D1_D3_D5_500.jsonl
│
├── position_begin_middle_end_500.jsonl
│
├── counterfactual_high_confidence_fixed.jsonl
│
├── counterfactual_skipped_fixed.jsonl
│
├── hard_negative_audit_500.jsonl
│
├── quality_report.json
│
├── counterfactual_quality_report.json
│
└── README.md
```

The older files:

```text
counterfactual_high_confidence.jsonl
counterfactual_skipped.jsonl
```

should be treated as legacy artifacts and should not be used for final experiments.

---

# 9. Loading the Datasets

```python
import json


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


clean = load_jsonl(
    "datasets/clean_500.jsonl"
)

distractor = load_jsonl(
    "datasets/distractor_D1_D3_D5_500.jsonl"
)

position = load_jsonl(
    "datasets/position_begin_middle_end_500.jsonl"
)

counterfactual = load_jsonl(
    "datasets/counterfactual_high_confidence_fixed.jsonl"
)
```

---

# 10. Recommended Evaluation Protocol

Use the same model, system prompt, decoding configuration, tokenizer, and maximum generation length across all comparable conditions.

A minimal QA prompt is:

```text
Answer the question using only the supplied context.

Return only the short answer.
Do not provide an explanation.

Context:
{context}

Question:
{question}

Answer:
```

For deterministic evaluation, use greedy decoding where supported:

```python
do_sample = False
temperature = 0
```

## Clean / Distractor / Position

Evaluate using:

- Exact Match
- Token F1

## Counterfactual

Evaluate using:

- `GOLD_FOLLOW`
- `COUNTERFACTUAL_FOLLOW`
- `OTHER`

and compute Counterfactual Follow Rate.

---

# 11. Intended Experimental Comparisons

### Distractor robustness

```text
Clean
  ↓
D1
  ↓
D3
  ↓
D5
```

Measure degradation as retrieval noise increases.

### Position robustness

```text
Beginning vs Middle vs End
```

Measure sensitivity to evidence placement.

### Counterfactual robustness

```text
Original dataset truth
        vs
Injected retrieved claim
```

Measure whether the model follows the provided context or preserves the original answer.

---

# 12. Reproducibility Notes

The dataset was intentionally constructed so that the perturbation under study is isolated as much as possible:

- distractor experiment varies hard-negative count
- position experiment keeps chunks fixed and changes only their order
- counterfactual experiment alters the target fact while retaining supporting-context structure
- excluded counterfactual examples are logged rather than silently discarded

Before running LLM inference, the final datasets should pass the corresponding dataset-validation notebook.

---

# Dataset Summary

```text
Source questions:                 500

Clean examples:                   500

Distractor:
  D1                              500
  D3                              500
  D5                              500
  Total                         1,500

Position:
  Beginning                       500
  Middle                          500
  End                             500
  Total                         1,500

Final counterfactual examples:    431
Counterfactual skipped:            69

Selected hard negatives:        2,500
Minimum semantic score:        0.3504
Mean semantic score:           0.7017
Median semantic score:         0.7103
```

---

## Project

**RAG-Stress** — Empirical stress-testing of retrieval-augmented generation under controlled retrieval/context failure conditions.
