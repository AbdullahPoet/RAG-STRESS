RAG-STRESS FINAL PERTURBATION DATASET

Input:
  500 HotpotQA questions supplied by the user.
  yes/no questions were already excluded.

Files:
  clean_500.jsonl
      500 clean examples

  distractor_D1_D3_D5_500.jsonl
      1500 examples
      500 D1 + 500 D3 + 500 D5

  position_begin_middle_end_500.jsonl
      1500 examples
      500 beginning + 500 middle + 500 end

  counterfactual_high_confidence.jsonl
      439 quality-screened counterfactual examples

Hard-negative rules:
  - Gold answer absent from every distractor
  - Supporting evidence excluded
  - Semantic score >= 0.35
  - 0.45 * question cosine + 0.55 * gold-context cosine
  - LSA latent-semantic embeddings
  - Question-specific HotpotQA distractors prioritized
  - Non-supporting sentences from gold documents used as hard topical negatives
  - Global fallback and answer-masked fallback only when necessary

Quality:
  Minimum selected score: 0.3504
  Mean selected score: 0.7017
  Median selected score: 0.7103
