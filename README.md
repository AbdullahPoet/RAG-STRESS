# RAG Stress Test: Evaluating LLM Robustness Under Retrieval Failure

I built this project to study a simple but important question:

> **How robust is an LLM when the retrieved context in a RAG system is noisy, badly positioned, or confidently wrong?**

Most RAG evaluations report performance when retrieval works correctly. In practice, retrieval can fail in several different ways. A retriever may return highly similar but irrelevant passages, the useful evidence may be buried in the middle of a long context, or the retrieved evidence may itself contain incorrect information.

In this project, I deliberately stress the retrieved context and measure how two Qwen models behave under those failures:

- **Qwen3-4B-Instruct-2507**
- **Qwen3.5-9B**

The goal is not only to compare raw accuracy. I want to understand **how model scale changes robustness**, which perturbations cause failures, and whether a model that is better at ignoring irrelevant context is also better at resisting false context.

---

## What I Did

I created a controlled RAG stress-testing pipeline with four evaluation settings:

```text
Clean Context
     |
     +-----------------------------+
     |                             |
     v                             v
Topical Distractors          Evidence Position
D1 / D3 / D5             Beginning / Middle / End
     |
     +-----------------------------+
     |
     v
Counterfactual Evidence
```

I first ran a clean baseline and then modified only the supplied context while keeping the question-answering setup fixed.

I tested three major RAG failure modes:

1. **Topical distractors**
   - I inserted semantically related passages that do not contain the answer.
   - I tested increasingly difficult settings with 1, 3, and 5 distractors.

2. **Position perturbation**
   - I moved the same answer-bearing evidence to the beginning, middle, and end of the context.
   - This tests whether the model suffers from a lost-in-the-middle style failure.

3. **Counterfactual injection**
   - I intentionally modified relevant context so that it supports an incorrect answer.
   - This tests whether the model blindly follows retrieved evidence when the evidence itself is wrong.

I then evaluated the same benchmark with a **4B model** and a **9B model** so that I could compare the effect of model scale.

---

# Why I Did This

A RAG system can fail even when the language model itself is strong.

A simplified RAG pipeline looks like this:

```text
Question
   |
   v
Retriever
   |
   v
Retrieved Context
   |
   v
LLM
   |
   v
Answer
```

The final answer therefore depends heavily on the quality of the retrieved context.

I wanted to separate three different questions:

### 1. Can the model ignore irrelevant but convincing context?

A retriever often returns passages that are topically similar to the query but do not actually contain the answer.

That motivated the **D1 / D3 / D5 distractor experiment**.

### 2. Does the location of evidence matter?

Long-context models do not necessarily use every token equally well.

That motivated the **Beginning / Middle / End position experiment**.

### 3. What happens when retrieval is relevant but factually wrong?

This is more dangerous than ordinary retrieval noise because the context can look useful while containing false evidence.

That motivated the **counterfactual experiment**.

Together, these experiments let me distinguish:

```text
Retrieval Noise Robustness
        vs.
Retrieval Truth Robustness
```

---

# Models

| Model | Approx. Parameters | Role |
|---|---:|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | Smaller baseline model |
| `Qwen/Qwen3.5-9B` | 9B | Larger comparison model |

The 9B model has roughly **2.25× more parameters** than the 4B model.

I kept the experimental setup as consistent as possible across the two models:

- same questions,
- same perturbation datasets,
- same benchmark prompt,
- same short-answer requirement,
- deterministic/greedy decoding,
- same answer normalization,
- same EM/F1 definitions,
- same counterfactual behavior definitions.

This matters because changing the prompt or evaluation procedure between models would confound the comparison.

---

# Experimental Pipeline

## 1. Clean Baseline

I first evaluated each model using the unmodified context.

This gives me the reference performance that every perturbation can be compared against.

```text
Question + Correct Context
          |
          v
         LLM
          |
          v
      Clean Answer
```

The clean baseline is important because robustness is not only about final accuracy.

I also measure:

> **How much performance does a model lose relative to its own clean baseline?**

---

## 2. Topical Distractor Perturbation

I added semantically related passages that do not contain the answer.

I evaluated three levels:

```text
D1 = 1 distractor
D3 = 3 distractors
D5 = 5 distractors
```

The basic idea is:

```text
Question
   |
   v
Correct Evidence
+
Semantically Similar Distractors
   |
   v
LLM
   |
   v
Answer
```

### Why I tested this

Embedding-based retrieval does not guarantee that every highly ranked chunk is useful.

A model used inside a real RAG system should be able to separate:

```text
relevant evidence
```

from:

```text
topically similar noise
```

---

## 3. Position Perturbation

I used the same answer-bearing evidence but changed where it appeared in the context.

The three conditions are:

```text
Beginning
Middle
End
```

For example:

```text
Beginning
[CORRECT] [NOISE] [NOISE] [NOISE]

Middle
[NOISE] [CORRECT] [NOISE] [NOISE]

End
[NOISE] [NOISE] [NOISE] [CORRECT]
```

### Why I tested this

If the answer changes only because the same evidence appears in a different location, then the model has position sensitivity.

I was particularly interested in the **middle condition** because long-context models can show a lost-in-the-middle effect.

---

## 4. Counterfactual Perturbation

This is the most adversarial part of the benchmark.

Instead of adding irrelevant information, I altered relevant evidence so that the context supports a deliberately incorrect answer.

Conceptually:

```text
Original fact:
X is the answer.

Counterfactual context:
Y is the answer.
```

The question itself stays the same.

I then observe whether the model:

```text
follows Y
follows X
or produces something else
```

I classify the outputs as:

- **COUNTERFACTUAL_FOLLOW**
- **GOLD_FOLLOW**
- **OTHER**

From these classes, I calculate:

- **Counterfactual Follow Rate (CFR)**
- **Gold Follow Rate (GFR)**
- **OTHER rate**

### Why I tested this

Distractor robustness and misinformation robustness are not the same thing.

A model can successfully ignore irrelevant passages while still trusting false retrieved evidence.

That distinction became one of the main findings of this project.

---

# Evaluation

For normal QA conditions I use:

### Exact Match

Exact Match checks whether the normalized predicted answer exactly matches the gold answer.

```text
EM = exact normalized prediction match
```

### Token-Level F1

F1 captures partial token overlap between the prediction and gold answer.

This is useful when a model produces a partially correct answer that does not satisfy strict Exact Match.

### Performance Drop

For each perturbation I also calculate the loss relative to clean performance:

```text
Drop = Clean Score - Perturbed Score
```

A smaller drop means better robustness.

---

# Statistical and Failure-Mode Analysis

I did not want to rely only on average EM/F1.

I also performed paired and question-level analysis using:

- paired bootstrap confidence intervals,
- McNemar tests for Exact Match,
- Wilcoxon signed-rank tests for per-example F1,
- clean-correct → perturbed-wrong transitions,
- robustness trajectory analysis,
- position transition patterns,
- strict lost-in-the-middle cases,
- middle-specific failure rates,
- counterfactual behavior by type,
- qualitative failure examples.

This lets me study not only:

> "Which model has the higher average score?"

but also:

> "Which individual questions fail when the context is perturbed?"

---

# Results

# 1. Clean Baseline

The larger model starts with a much stronger clean baseline.

| Model | Exact Match | F1 |
|---|---:|---:|
| Qwen3-4B | 49.20% | 66.57% |
| Qwen3.5-9B | **62.00%** | **77.51%** |
| Difference | **+12.80 pp** | **+10.94 pp** |

### What I observed

Moving from 4B to 9B parameters gives a large improvement even before any perturbation is applied.

The 9B model improves:

```text
EM: 49.20% -> 62.00%
F1: 66.57% -> 77.51%
```

However, clean accuracy alone does not tell me whether the model is robust.

That becomes clearer in the perturbation experiments.

---

# 2. Perturbation-Wise Comparison

## 2.1 Topical Distractors

### Exact Match

| Condition | Qwen3-4B | Qwen3.5-9B |
|---|---:|---:|
| Clean | 49.20% | **62.00%** |
| D1 | 46.20% | **60.60%** |
| D3 | 43.40% | **60.20%** |
| D5 | 43.00% | **60.60%** |

The 4B trajectory is:

```text
49.2 -> 46.2 -> 43.4 -> 43.0
```

The 9B trajectory is:

```text
62.0 -> 60.6 -> 60.2 -> 60.6
```

### F1

| Condition | Qwen3-4B | Qwen3.5-9B |
|---|---:|---:|
| Clean | 66.57% | **77.51%** |
| D1 | 61.25% | **76.57%** |
| D3 | 59.29% | **75.18%** |
| D5 | 57.97% | **75.75%** |

### Drop from Clean

| Condition | 4B EM Drop | 9B EM Drop | 4B F1 Drop | 9B F1 Drop |
|---|---:|---:|---:|---:|
| D1 | 3.00 pp | **1.40 pp** | 5.32 pp | **0.94 pp** |
| D3 | 5.80 pp | **1.80 pp** | 7.29 pp | **2.33 pp** |
| D5 | 6.20 pp | **1.40 pp** | 8.61 pp | **1.76 pp** |

### Clean-Correct → Perturbed-Wrong

| Condition | Qwen3-4B | Qwen3.5-9B |
|---|---:|---:|
| D1 | 9.35% | **4.52%** |
| D3 | 15.85% | **7.42%** |
| D5 | 16.26% | **7.42%** |

## What I found

This is one of the clearest outcomes of the project.

The 4B model becomes progressively weaker as distractor load increases.

The 9B model remains almost flat.

At D5:

```text
4B EM drop = 6.20 percentage points
9B EM drop = 1.40 percentage points
```

The clean-correct → D5-wrong transition rate also drops from:

```text
16.26% -> 7.42%
```

when moving from 4B to 9B.

### My interpretation

Increasing model capacity substantially improves the model's ability to ignore irrelevant but semantically similar retrieved passages.

In other words:

> **The 9B model is much more robust to retrieval noise.**

---

# 2.2 Position Perturbation

| Position | Qwen3-4B EM | Qwen3.5-9B EM | Qwen3-4B F1 | Qwen3.5-9B F1 |
|---|---:|---:|---:|---:|
| Beginning | 43.20% | **60.00%** | 59.10% | **74.69%** |
| Middle | 42.00% | **58.20%** | 58.29% | **72.84%** |
| End | 44.40% | **59.80%** | 60.60% | **75.21%** |

### Drop from Clean

| Position | 4B EM Drop | 9B EM Drop |
|---|---:|---:|
| Beginning | 6.00 pp | **2.00 pp** |
| Middle | 7.20 pp | **3.80 pp** |
| End | 4.80 pp | **2.20 pp** |

For both models, the middle condition is the weakest.

```text
4B middle EM = 42.0%
9B middle EM = 58.2%
```

I also measured the probability that the middle fails when at least one edge position succeeds.

| Model | Middle Failure Given Edge Success |
|---|---:|
| Qwen3-4B | 15.16% |
| Qwen3.5-9B | **7.64%** |

## What I found

Both models show position sensitivity.

However, the larger model reduces the effect substantially.

The conditional middle-failure rate is approximately halved:

```text
15.16% -> 7.64%
```

### My interpretation

Scaling does not completely remove lost-in-the-middle behavior.

It does make the model substantially less sensitive to where the answer-bearing evidence appears.

So my conclusion is:

> **The 9B model has better positional robustness, but position remains a real source of error.**

---

# 2.3 Counterfactual Perturbation

For the final comparison, I use the same strict **Counterfactual V2** protocol for both models.

Each model is evaluated on **232 counterfactual examples**.

| Model | Counterfactual Follow | Gold Follow | OTHER |
|---|---:|---:|---:|
| Qwen3-4B | 48.28% | 3.45% | 48.28% |
| Qwen3.5-9B | **59.48%** | 3.02% | **37.50%** |

The Counterfactual Follow Rate changes from:

```text
48.28% -> 59.48%
```

That is an increase of:

```text
+11.20 percentage points
```

## What I found

This result goes in the opposite direction from the distractor experiment.

The 9B model is much better at ignoring irrelevant context, but when the supplied evidence itself is false, it follows that evidence **more often**.

The larger model also produces fewer OTHER responses:

```text
48.28% -> 37.50%
```

This suggests that it is more decisive about the information contained in the supplied context.

### Important interpretation

A higher Counterfactual Follow Rate is **not a robustness improvement**.

In this experiment, it means that the model follows deliberately manipulated evidence more frequently.

Therefore:

> **The 9B model is more robust to irrelevant context, but more susceptible to relevant-looking false context under this counterfactual setup.**

---

# 3. Model Parameter-Wise Comparison

The second way I analyze the experiment is by model scale.

```text
Qwen3-4B
   |
   |  ~2.25x parameters
   v
Qwen3.5-9B
```

## Clean Performance

| Metric | 4B | 9B | Change |
|---|---:|---:|---:|
| EM | 49.20% | **62.00%** | +12.80 pp |
| F1 | 66.57% | **77.51%** | +10.94 pp |

The larger model is clearly stronger before perturbation.

---

## Distractor Robustness

At D5:

| Metric | 4B | 9B |
|---|---:|---:|
| EM | 43.00% | **60.60%** |
| EM Drop | 6.20 pp | **1.40 pp** |
| F1 | 57.97% | **75.75%** |
| F1 Drop | 8.61 pp | **1.76 pp** |
| Clean-correct → wrong | 16.26% | **7.42%** |

This is a strong scaling benefit.

---

## Position Robustness

For the hardest middle condition:

| Metric | 4B | 9B |
|---|---:|---:|
| Middle EM | 42.00% | **58.20%** |
| EM Drop | 7.20 pp | **3.80 pp** |
| Middle F1 | 58.29% | **72.84%** |
| Middle fail given edge success | 15.16% | **7.64%** |

Again, the larger model is substantially more robust.

---

## Counterfactual Susceptibility

| Metric | 4B | 9B | Change |
|---|---:|---:|---:|
| CFR | 48.28% | **59.48%** | +11.20 pp |
| GFR | 3.45% | 3.02% | -0.43 pp |
| OTHER | 48.28% | **37.50%** | -10.78 pp |

This reveals a different effect of scaling.

The larger model appears to make stronger use of the supplied evidence.

That is useful when retrieval is correct.

It can become a weakness when retrieval is wrong.

---

# Main Outcome

The central result of my project is not simply:

> "9B is better than 4B."

The more interesting result is:

```text
More Parameters
      |
      +--> Better clean QA
      |
      +--> Better distractor robustness
      |
      +--> Better position robustness
      |
      +--> Stronger use of supplied context
                    |
                    +--> Good when context is correct
                    |
                    +--> Risky when context is false
```

My experiments therefore show that robustness is **multidimensional**.

A model can improve on one type of RAG failure while becoming more vulnerable to another.

---

# Final Comparison

| Property | Qwen3-4B | Qwen3.5-9B |
|---|---|---|
| Clean QA performance | Lower | **Higher** |
| Robustness to D1/D3/D5 distractors | Moderate | **Strong** |
| Performance stability as distractors increase | Lower | **Higher** |
| Position robustness | Lower | **Higher** |
| Middle-position sensitivity | Higher | **Lower** |
| Clean-correct preservation | Lower | **Higher** |
| Counterfactual follow rate | **Lower** | Higher |
| Resistance to injected false evidence in this benchmark | **Better** | Weaker |
| Decisiveness under supplied context | Lower | **Higher** |

---

# Key Numbers

```text
Clean EM
4B: 49.20%
9B: 62.00%

D5 EM
4B: 43.00%
9B: 60.60%

D5 EM degradation
4B: 6.20 pp
9B: 1.40 pp

Middle EM
4B: 42.00%
9B: 58.20%

Middle failure given edge success
4B: 15.16%
9B: 7.64%

Counterfactual Follow Rate
4B: 48.28%
9B: 59.48%
```

---

# What I Learned

This project changed how I think about RAG robustness.

At first, it is tempting to define robustness as:

```text
"Does accuracy remain high after adding noise?"
```

But the experiments show that this definition is incomplete.

A production RAG system needs at least two different kinds of robustness.

### 1. Relevance Robustness

Can the model ignore information that is related to the topic but irrelevant to the answer?

The 9B model performs very well here.

### 2. Truth Robustness

Can the model question evidence that looks relevant but is actually false?

The counterfactual experiment shows that this remains difficult.

This leads to the main research implication from my project:

> **Better context utilization is not automatically better context verification.**

A stronger model may be better at finding and following evidence while still requiring an external mechanism to determine whether that evidence should be trusted.

---

# Why This Matters for Real RAG Systems

A real-world retriever can fail in multiple ways:

```text
Retriever Failure
      |
      +--> Irrelevant retrieval
      |
      +--> Too many similar chunks
      |
      +--> Important evidence buried in context
      |
      +--> Stale information
      |
      +--> Incorrect information
      |
      +--> Poisoned retrieval source
```

My results suggest that increasing model size helps significantly with the first group of problems.

It does not automatically solve the second group.

A more reliable RAG architecture may therefore need:

```text
Retriever
   |
   v
Relevance Filtering
   |
   v
Evidence Verification
   |
   v
LLM
   |
   v
Answer
```

rather than relying on the LLM alone to determine whether retrieved evidence is trustworthy.

---

# Experimental Sizes

| Experiment | Qwen3-4B | Qwen3.5-9B |
|---|---:|---:|
| Clean | 500 | 500 |
| Distractor | 1,500 | 1,500 |
| Position | 1,500 | 1,500 |
| Counterfactual V2 | 232 | 232 |

The distractor experiment contains three 500-example conditions:

```text
D1
D3
D5
```

The position experiment also contains three 500-example conditions:

```text
Beginning
Middle
End
```

---

# Implementation Notes

For inference, I used a fixed short-answer benchmark prompt and deterministic generation.

The 4B pipeline includes batched and resumable inference designed for GPU execution.

For Qwen3.5-9B, I ran the model in BF16 on an NVIDIA L4 without 4-bit or 8-bit quantization.

The perturbation files already contain the complete contexts required for each experiment, so inference does not rebuild a vector database. This is intentional: the purpose of this stage is to isolate **LLM behavior under controlled retrieved contexts**, not to measure retriever variability.

Outputs are saved as JSONL so experiments can be resumed and evaluated independently.

---

# Repository Workflow

The notebooks follow the experiment from inference through statistical analysis.

```text
Perturbation datasets
        |
        v
LLM inference
        |
        v
Raw JSONL predictions
        |
        v
EM / F1 / Counterfactual evaluation
        |
        v
Paired statistical analysis
        |
        v
Failure-mode analysis
        |
        v
4B vs 9B comparison
```

The main notebooks include:

```text
01_run_llm_inference.ipynb
02_evaluate_rag_stress.ipynb
02b_fix_counterfactual_inference.ipynb
02c_counterfactual_v2_infer_evaluation.ipynb
03_qwen3_4b_analysis.ipynb
01_qwen35_9b_L4_full_inference_evaluation.ipynb
02_qwen35_9b_statistical_analysis.ipynb
in the notebooks directory
```

---

# Research Questions

I designed the project around three main research questions.

### RQ1 — Distractor Robustness

> Does performance degrade as the number of semantically related distractors increases?

**Outcome:** Yes for the 4B model, but the 9B model is much more stable.

---

### RQ2 — Position Sensitivity

> Does moving the same evidence to the beginning, middle, or end change performance?

**Outcome:** Yes. Both models perform worst in the middle, but the 9B model is less sensitive.

---

### RQ3 — Counterfactual Susceptibility

> When retrieved evidence supports a false answer, does the model follow the retrieved evidence or resist it?

**Outcome:** The 9B model follows the injected counterfactual evidence more often than the 4B model.

---

# Conclusion

I built this project to move beyond normal RAG accuracy evaluation and directly test what happens when retrieved context fails.

The experiments show that increasing model size from 4B to 9B provides clear benefits:

- higher clean QA accuracy,
- much stronger distractor resistance,
- smaller performance degradation,
- better preservation of clean-correct answers,
- and lower sensitivity to evidence position.

However, I also found an important limitation.

The larger model follows deliberately false retrieved evidence more frequently.

My final conclusion is:

> **Scaling makes the model better at using context, but being better at using context is not the same as being better at verifying context.**

For reliable RAG systems, retrieval relevance and evidence truthfulness should therefore be evaluated separately.

This project demonstrates why a complete RAG robustness benchmark should include not only clean accuracy and distractor tests, but also **position perturbation and counterfactual evidence injection**.
