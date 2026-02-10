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

## Key Insights
- Student has strong real-world production ML experience
- Good at building frameworks but sometimes imprecise on mathematical details
- Responds well to concrete examples (fraud detection scenario was very effective)
- Segmentation insight for operational monitoring shows engineering maturity

## Follow-up Topics
- C.19 CNNs (next on study plan)
- Review BatchNorm/LayerNorm details
- Continue class imbalance gradient derivation
