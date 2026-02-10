# Session Notes - 2026-02-09

## Session Overview
- **Date**: 2026-02-09
- **Topics Covered**: E.36 Monitoring & Model Degradation, C.18 BatchNorm vs LayerNorm
- **Session Type**: Guided learning (continue study plan)

---

## Topic 1: E.36 Monitoring and Model Degradation

### Student's Baseline
- Knew data drift and model degradation concepts
- Understood feature distribution shift, label distribution shift, model distribution shift
- Had real-world experience: deployed models with monitoring, used reverse A/B testing
- Strong practical foundation

### Concepts Taught
1. **Three types of drift** — covariate shift, label shift, concept drift
2. **Label shift vs concept drift distinction** — student initially couldn't differentiate
   - Used fraud detection example: "same fraud, different amount" vs "different fraud altogether"
   - Student successfully diagnosed concept drift scenario (fraudsters changing tactics)
3. **Four-layer monitoring framework** — built interactively:
   - Data monitoring (feature distributions, label distributions, schema, nulls)
   - Model monitoring (prediction distribution shifts, confidence changes)
   - Operational monitoring (error rates, latency, memory/OOM — segmented by pipeline step)
   - Business monitoring (CTR, session depth, DAU, retention vs baseline)
4. **Alert tiering** — P0 (immediate page), P1 (hours), P2 (daily review)
5. **Concept drift prevention** — scheduled retraining, online learning, human-in-the-loop, output distribution monitoring

### Comprehension Checks
- ✅ Correctly identified concept drift in fraud scenario
- ✅ Proposed valid solutions: retrain with new data + new features
- ✅ Built comprehensive monitoring framework unprompted
- ✅ Strong operational monitoring with segmentation insight
- ✅ Good business metrics (CTR, DAU, retention with baseline comparison)

### Assessment
- **Confidence Level**: Medium-High
- **Status**: Mastered — has interview-ready framework
- **Knowledge Gap Resolved**: "Monitoring & model degradation" gap filled

---

## Topic 2: C.18 Batch Normalization vs Layer Normalization

### Student's Baseline
- Knew BatchNorm for vision, LayerNorm for NLP
- Understood variable sentence length argument for LayerNorm
- Knew about RMSNorm existence
- Confused L1/L2 regularization with normalization techniques (corrected)

### Concepts Taught
1. **Regularization vs normalization** — different purposes (penalty vs activation stabilization)
2. **Exact mechanics** — BatchNorm normalizes across batch dim, LayerNorm across feature dim
3. **Deeper reason for LayerNorm in NLP** — batch dependency, train/test discrepancy with running statistics
4. **RMSNorm correction** — removes mean centering + beta, not variance; ~10-15% faster
5. **BatchNorm placement** — before activation (original paper), not after
6. **Pre-Norm vs Post-Norm Transformers** — Pre-Norm trains more stably (GPT uses this)

### Comprehension Checks
- ✅ Correctly explained padding token pollution
- ❌ Thought BatchNorm goes after activation (corrected)
- ⚠️ Slightly inaccurate on RMSNorm mechanics (corrected)

### Assessment
- **Confidence Level**: Medium
- **Status**: Partially covered — understands concepts but had detail inaccuracies
- **Needs Review**: BatchNorm placement, RMSNorm precise mechanics

---

## Quiz Session: Practical ML + Mathematical Deep Dive

### Topic: Class Imbalance — Weighted Cross-Entropy
- Student chose: Practical ML (8) + Mathematical Deep Dive (2)
- Generated topic: Modifying cross-entropy for 98%/2% class imbalance
- Student wrote BCE formula (missed negative sign — corrected)
- Student proposed weighting scheme using (1 - class_frequency) — good intuition but mixed up positive/negative classes
- Session ongoing — gradient derivation in progress

---

---

## Topic 3 (Quiz): F.39 Handling Imbalanced Data + Weighted Cross-Entropy

### Student's Baseline
- Knew BCE formula (missed negative sign — corrected)
- Knew about weighted cross-entropy and focal loss concepts

### Concepts Taught
1. **Weighted cross-entropy** — derived formula, gradient analysis showing 49x larger gradient for minority class
2. **Focal loss** — (1-ŷ)^γ modulating factor, γ=0 reduces to weighted CE, from RetinaNet paper (Lin et al., 2017)
3. **Other approaches** — oversampling/SMOTE, undersampling, data augmentation, threshold tuning
4. **Connected to AUC-ROC** — why AUC handles imbalance (TPR/FPR normalized within class)

### Comprehension Checks
- ✅ Correctly identified focal loss reduces to weighted CE when γ=0
- ✅ Proposed valid imbalance solutions (sampling, augmentation)
- ⚠️ Mixed up positive/negative classes initially in weighting (corrected)
- ⚠️ Misinterpreted AUC=0.92 as "92% correct on the class" (corrected — it's ranking probability)

### Assessment
- **Confidence Level**: Medium-High
- **Status**: Solid understanding of techniques, minor interpretation issues corrected

---

## Topic 4 (Review): A.8 AUC-ROC Reinforcement + AUC-PR

### Concepts Reviewed/Taught
1. **AUC-ROC axes** — student swapped axes again (same as Session 1). FPR on x, TPR on y. Mnemonic: F=First(x), T=Top(y)
2. **Probabilistic interpretation** — correctly restated: "AUC of x means x chance positive scores higher than negative"
3. **How AUC is calculated** — trapezoidal rule, threshold sweeping
4. **Why random = 0.5** — coin flip on ranking
5. **AUC-PR** — precision doesn't involve TN, so not diluted by massive TN in imbalanced data
6. **AUC-PR baseline** — equals positive class rate (0.02), not 0.5. Judge by magnitude of improvement

### Comprehension Checks
- ✅ Excellent final summary of AUC-ROC vs AUC-PR comparison
- ⚠️ Swapped ROC axes again — needs reinforcement
- ⚠️ Initially said AUC-PR 0.03 vs 0.02 is "strong improvement" (corrected — only 1.5x)
- ⚠️ Said AUC-PR "uses fn to replace tn" (corrected — precision simply doesn't involve TN)

### Assessment
- **Confidence Level**: Medium-High (improved from Session 1, but axes still need drilling)

---

## Topic 5 (Quiz): F.42 SHAP & Model Interpretability

### Student's Baseline
- No prior knowledge of SHAP

### Concepts Taught
1. **Shapley values from game theory** — feature contribution = marginal impact averaged over all orderings
2. **Weighting formula** — |S|!×(|F|-|S|-1)!/|F|! explained via ordering counting
3. **Worker building a house analogy** — "joining first/second/third" = position in arrival order
4. **Computational complexity** — O(n!), intractable for real models
5. **Practical approximations** — sampling, TreeSHAP O(TLD²), KernelSHAP, DeepSHAP
6. **SHAP vs feature importance** — local vs global, directionality, theoretical guarantees (efficiency, symmetry, null player)

### Comprehension Checks
- ✅ Understood n! complexity
- ✅ Understood γ=0 focal loss connection
- ✅ Gave strong interview-ready summary of SHAP

### Assessment
- **Confidence Level**: Medium — understood concepts well, but brand new topic needs practice
- **Status**: Covered fundamentals, could dive deeper into TreeSHAP mechanics

---

## Key Insights
- Student has strong real-world production ML experience
- Good at building frameworks but sometimes imprecise on mathematical details
- Responds well to concrete examples (fraud detection scenario, house-building analogy)
- Segmentation insight for operational monitoring shows engineering maturity
- ROC axes confusion is recurring — needs targeted drilling
- Strong at connecting concepts across topics when guided

---

## Topic 6: F.38 Debugging Training Issues

### Student's Baseline
- Had practical experience: restart from checkpoint, lower learning rate, gradient clipping
- Knew to monitor loss during training
- Good instincts but approach was unsystematic

### Concepts Taught
1. **Debugging hierarchy** — structured order: data → sanity checks → training mechanics → regularization
2. **Initial loss sanity check** — loss should be log(k) for k classes; for LLMs with vocab 32k, expect ~10.4
3. **Loss-to-probability conversion** — ŷ = e^(-loss), useful for interpreting LLM training progress
4. **Overfit tiny batch** — single most powerful diagnostic, validates entire pipeline end-to-end
5. **NaN debugging** — specific causes: log(0), exp overflow, 0/0 derivatives, data NaN, exploding gradients
6. **Sudden NaN at step N** — weights grow over time, model overconfidence, bad data batch, LR schedule change
7. **Overfitting vs underfitting** — opposite diagnoses, opposite fixes
8. **L1 = Lasso** — corrected student listing them as separate techniques

### Comprehension Checks
- ✅ Understood initial loss sanity check and asked great follow-up about LLM vocab sizes
- ✅ Good question about loss-to-probability conversion (why e⁻³)
- ✅ Correctly identified overfitting scenario and fixes
- ✅ Correctly identified underfitting and opposite fixes
- ⚠️ Didn't initially think of "overfit tiny batch" as first diagnostic
- ⚠️ Listed L1/L2/Lasso as three separate things (corrected)
- ⚠️ Didn't fully explain why NaN appears suddenly (needed guidance on weight growth / overconfidence)

### Assessment
- **Confidence Level**: Medium-High
- **Status**: Strong practical foundation, now has systematic interview framework
- **Key win**: 4-row debugging summary table (overfitting, underfitting, not learning, NaN)

---

## Follow-up Topics
- C.19 CNNs (next on study plan)
- Review BatchNorm/LayerNorm details
- Drill ROC axes (recurring confusion)
- Practice SHAP explanations
- TreeSHAP deeper dive
- A.5 Bias-variance tradeoff (naturally follows from overfitting/underfitting discussion)
