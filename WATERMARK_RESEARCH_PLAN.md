# Watermark Research Plan

## Project Overview
- **Context**: This repository experiments with watermarking a diffusion-based language model (LLaDA) by biasing the remasking/unmasking steps during generation. The watermark is a deterministic, keyed pair-rule that nudges token selection toward tokens satisfying the rule.
- **Mechanism**: `SimpleRemaskWatermark` uses a hash-based deterministic `pair_rule_follows` mapping between token pairs and a binary decision. During remasking, it biases the model's confidence for tokens that satisfy the pair rule by a configurable delta (`WATERMARK_DELTA`), and the detection code counts satisfied pair occurrences in generated text and performs a binomial (and z-score) analysis to derive a detection statistic.
- **Why**: Adds ownership and provenance signals to model outputs while aiming to minimize quality degradation and performance overhead.

## Key Metrics & Datasets
- **Datasets**: ELI5 (question prompts), C4, and synthetic/null sampling (for FPR/false-positive calibration).
- **Detection metrics**: True Positive Rate (TPR), False Positive Rate (FPR), z-score and p-value distribution, satisfaction rate across contexts, detection margin (threshold - p_value).
- **Quality metrics**: Perplexity (using GPT-2 baseline), cosine similarity (all-MiniLM-L6-v2), lexical statistics (avg sentence/word length, lexical diversity, Flesch reading ease), and sequence length.
- **Performance metrics**: Generation time and time overhead of watermarking.

## Primary Research Objectives
- Implement and evaluate a watermarking method for diffusion LMs that:  
  - Achieves high detectability (high TPR) with low FPR,  
  - Minimizes text quality degradation (acceptable perplexity and semantic similarity),  
  - Maintains acceptable generation performance (minimized run-time overhead).
- Provide a reliable and calibrated detection framework:  
  - Define null models and quantify the distribution of satisfied pairs under no-watermark generation,  
  - Compute p-values and z-scores for detection and set operational thresholds for FPR control.
- Investigate different methods of biasing the generation process, including direct logit modification prior to softmax, to more closely align with theoretically grounded approaches.

## Evaluation & Validation Goals
- Characterize detection performance and utility trade-offs across:  
  - Watermark strength (`delta`) and key choices,  
  - Generation lengths and remasking parameters (steps, block_length),  
  - Different datasets and prompt distributions, and  
  - Remasking strategies (low-confidence vs random) and sampling temperatures.
- Calibrate FPR by sampling non-watermarked outputs and verify null hypothesis distribution (scripts: `find_null.py`).
- Report detection strength (mean z-scores and detection percent at defined thresholds) across experiment settings.

## Robustness & Stress Tests
- Evaluate watermark resilience to common obfuscations: paraphrasing, synonym replacement, truncation, re-tokenization, sampling noise, and paraphrase-based re-generation across different models.
- Test detection stability under parameter variation: `delta`, `p_value_threshold`, `block_length`, and remasking choices.
- Verify the effect of adversarial or adaptive attacks (e.g., paraphrase, sequence editing) on detection rates and detection margin.

## Operationalization & Thresholds
- Use p-value thresholds (0.05/0.01/0.005) and z-score thresholds (z≥2.326 ~ 99%, z≥3.09 ~ 99.9%) as policies for declaring watermark presence.
- Establish recommended `delta` ranges given acceptable quality/performance trade-offs based on experiments.

## Suggested Experiments & Deliverables
- Reproducible experiments and scripts that:  
  - Sweep `delta`, remasking strategy, generation length, and sampling temperature to measure TPR/FPR and text quality impact,  
  - Build null distributions with large samples of non-watermarked outputs to calibrate FPR,  
  - Run robustness tests where watermarked outputs are paraphrased or passed through other models.
- Deliverables:  
  - `WATERMARK_RESEARCH_PLAN.md` (this file),  
  - Reproducible experiment scripts (if not already present),  
  - JSON and plot outputs collected in `runs/` (TPR/FPR vs generation length, detection z-score distribution, PPL increase vs delta),  
  - A short methods summary documenting `pair_rule_follows`, all parameters, and detection and calibration methodology.

## Open/Exploratory Questions
- How transferable is the watermark across milder text transformations (paraphrase, re-tokenization, translation)?
- Can we design an owner-verifiable watermarking variant (proofs/verification) that is robust and privacy-preserving?
- Can watermark strength be optimized for stealth (minimal perceptible change) while retaining calibrated detection power?
- Can we implement the extended Red-Green watermark from the "Watermarking Diffusion Language Models" paper and compare its performance to our pair-rule based approach?
- Can we formulate our watermarking scheme as a constrained optimization problem to better understand its theoretical properties?
- How could we incorporate a more probabilistic application of the watermark, for instance by considering the expected 'green-ness' over multiple possible contexts?

## Comparison to Existing Work (DiffusionLM-Watermark)
The paper "Watermarking Diffusion Language Models" proposes a watermarking scheme for DLMs that is conceptually similar to this project's approach. However, there are some key differences:
- **Watermarking Scheme**: The paper uses an "extended Red-Green" watermark. This method biases the distribution towards tokens that will make *other* tokens satisfy a rule (i.e., become "green"). Our current `SimpleRemaskWatermark` uses a simpler binary pair-rule, which biases a token's confidence based on its immediate neighbors satisfying the rule.
- **Theoretical Grounding**: The paper's method is presented as an optimal solution to a constrained optimization problem. Our current approach is more heuristic, based on directly modifying confidence scores.
- **Evaluation**: This project has a strong, established evaluation framework, which can be used to rigorously compare our method to the one proposed in the paper.

## Next Steps
- Run null-distribution experiments using `find_null.py` to calibrate baseline satisfaction rates and p-value thresholds.  
- Run sweep experiments for `delta` and generation lengths using `run_multi_length_eval.py` and `modgood_script.py`.  
- Add targeted robustness scripts (paraphrase/obfuscate outputs and retest detection).  
- Implement a version of the extended Red-Green watermark from the paper and compare its performance against the existing `SimpleRemaskWatermark`.


---
Generated on: 2025-12-13

