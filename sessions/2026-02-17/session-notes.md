# Session Notes: 2026-02-17

## Session Overview
- **Date**: 2026-02-17
- **Duration**: ~60 minutes
- **Main Topics**: ML System Design Cost Optimization, Quantization (Mathematical Deep Dive), Knowledge Distillation (Mathematical Deep Dive)
- **Session Type**: Conceptual + Mathematical

---

## Questions Asked

1. How to save cost in ML system design?
2. How does semantic caching work?
3. Why asymmetric quantization on activations?
4. Quantization details at mathematical level
5. How QAT simulates quantization during training, why retraining is required
6. Knowledge distillation at mathematical depth

---

## Topics Covered

### E.38 ML System Cost Optimization

**Initial Understanding**: Strong practical foundation. Correctly identified data storage and model serving as major cost centers. Had real-world experience with OOM issues and horizontal scaling. Also understood training costs (offline/online) and feature pipeline costs (batch/streaming).

**Explanation Given**: Built comprehensive cost optimization framework through Socratic questioning. Covered model compression (distillation, quantization, pruning), smart inference (cascading, speculative decoding, MoE), and infrastructure-level optimizations (caching, batching, auto-scaling).

**Comprehension Check**: "Your model's inference costs are 3x higher than expected. Cut serving costs by 50% without significantly hurting quality. Walk me through your approach."

**Student Response**: Gave well-structured answer prioritizing low-risk/low-effort first: (1) check monitors and right-size resources, (2) adopt scaling methods, (3) implement caching, (4) model cascading/routing, (5) quantization/distillation/pruning. Missed batching as an early win.

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Almost (needs to remember batching, and practice measuring after each optimization)

### Semantic Caching

**Initial Understanding**: None — student asked to learn about it.

**Explanation Given**: Embed queries with sentence transformer, store in vector DB, ANN search for similar queries, return cached response if above similarity threshold.

**Comprehension Check**: "What's the risk of setting similarity threshold too low vs too high?"

**Student Response**: Excellent — correctly identified false positives (wrong answers served) at low threshold, and excessive misses at high threshold. Then proactively asked how to determine the threshold.

**Follow-up Teaching**: Connected threshold tuning to precision-recall tradeoff (cache hit = positive prediction). Student independently mapped out TP/FP/FN/TN for cache decisions.

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Yes

### Quantization (Mathematical Deep Dive)

**Initial Understanding**: Knew the concept (mapping FP32 to lower precision), understood symmetric vs asymmetric distinction at high level. Minor imprecision: described FP32 as "0 to 2^32" range.

**Explanation Given**: Step-by-step derivation of uniform quantization:
- Scale factor: s = (r_max - r_min) / (2^b - 1)
- Zero-point: z = round(-r_min / s)
- Quantize: q = round(r / s) + z
- Dequantize: r_hat = s * (q - z)
- Worked through concrete example (r_min=-0.8, r_max=1.2, INT8)

**Comprehension Checks**:
1. "Quantize r=0.5 and dequantize back" — Student performed calculation (minor arithmetic slip: rounded 63.78 to 63 instead of 64), correctly identified quantization error (0.50176 vs 0.5)
2. "Why does symmetric quantization waste buckets?" — Excellent explanation of wasted range [-1.2, -0.8)
3. "Why asymmetric for activations?" — Correctly reasoned about ReLU outputs being >= 0, half the buckets wasted with symmetric
4. "When would symmetric be fine?" — Correctly identified tanh ([-1,1], centered around 0)

**Additional Topics Covered**:
- PTQ vs QAT tradeoffs
- Straight-Through Estimator (STE) for backprop through round()
- Student initially confused STE with stop gradient — corrected
- Student confused STE "identity function" with "indicator function" — corrected

**Interview Rehearsal**: Student gave comprehensive quantization summary covering: mapping concept, formulas, symmetric vs asymmetric, PTQ vs QAT. Missed STE and the symmetric-for-weights/asymmetric-for-activations rule.

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Almost (needs to remember STE and when-to-use-which rule)

### MoE Clarification

**Initial Understanding**: Had the right concept but conflated MoE with pruning/L1 regularization.

**Explanation Given**: MoE is an architectural design (gating network routes to subset of experts), not pruning. Saves compute but NOT memory (full model loaded). Student corrected their understanding and gave good example (8/256 experts = 3% compute).

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Yes (with nuance about memory vs compute)

### QAT Deep Dive

**Initial Understanding**: Knew QAT from earlier discussion but didn't know the mechanism.

**Explanation Given**: Fake quantization (quantize → dequantize round trip) applied to weights and activations during forward pass. Output stays in FP32 so training remains stable, but model "feels" quantization error. Backward pass uses STE. At deployment, remove fake quantization and use real INT8.

**Comprehension Checks**:
1. "What gets fake-quantized besides weights?" — Correctly identified activations. Also mentioned normalization layers (good thinking but clarified these stay FP32).
2. "What happens at deployment?" — Correct: remove fake quantization, quantize to INT8, serve normally.
3. "Why does PTQ fail at INT4?" — Correctly explained bucket size becomes too coarse (0.133 vs 0.00784).

**Interview Rehearsal**: Strong end-to-end QAT explanation covering PTQ's problem, fake quantization mechanism, what gets quantized, STE for backprop, deployment. Minor formula typo (wrote +r instead of +z). Used "fade" instead of "fake" a few times.

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Almost (fix terminology: "fake" not "fade", watch formula accuracy)

### Knowledge Distillation (Mathematical Deep Dive)

**Initial Understanding**: Knew teacher-student concept from cost optimization discussion. Good intuition about soft targets carrying more information.

**Explanation Given**:
- Dark knowledge: soft probabilities encode inter-class relationships (cat similar to dog, far from car)
- Temperature T softens distributions: p_i = e^(z_i/T) / Σe^(z_j/T)
- Concrete example showing T=1 (peaky) vs T=5 (smooth) distributions
- KL divergence as loss (equals 0 when distributions match perfectly)
- Combined loss: L = α·T²·KL(teacher||student) + (1-α)·CE(y, student)
- T² scaling: compensates for 1/T² gradient reduction from temperature scaling
- α controls balance: α=1 pure distillation (bounded by teacher), α=0 normal training

**Comprehension Checks**:
1. "Which carries more info: soft targets or hard labels?" — Correctly explained soft targets encode relationships between all classes
2. "How to smooth the distribution?" — Immediately connected to temperature from InfoNCE (Session 7)
3. "What loss to match distributions?" — Correctly suggested KL divergence and cross-entropy
4. "Why KL over CE?" — Understood KL=0 at perfect match, CE gives non-zero constant
5. "What does each loss term do?" — Clear explanation of L_distill vs L_student roles
6. "Why T² scaling?" — Correctly intuited that dividing by T introduces 1/T² in gradients

**Assessment**:
- Understanding Level: 4/5
- Interview Ready: Almost (didn't get to do full rehearsal — session ended)

---

## Knowledge Gaps Identified

| Gap | Severity | Notes |
|-----|----------|-------|
| Batching as early cost optimization | Low | Didn't mention in interview-style answer; knows the concept |
| STE in quantization summary | Medium | Forgot to include in rehearsal answer |
| Symmetric vs asymmetric usage rule | Low | Understands reasoning but didn't state the rule explicitly |
| Minor arithmetic (rounding) | Low | Rounded 63.78 to 63 instead of 64 |
| QAT terminology: "fake" not "fade" | Low | Said "fade quantization" multiple times |
| QAT formula accuracy | Low | Wrote q=round(r/s)+r instead of +z in rehearsal |
| Distillation rehearsal incomplete | Medium | Session ended before full interview rehearsal — do next session |

---

## Topics Mastered This Session

| Topic | Confidence | Evidence |
|-------|------------|----------|
| ML System Cost Optimization (comprehensive) | Medium-High | Covered model compression, smart inference, infrastructure optimization. Structured prioritization in interview answer |
| Quantization (mathematical) | Medium-High | Derived formulas, computed examples, explained tradeoffs between symmetric/asymmetric, PTQ/QAT, understood STE |
| QAT mechanism (fake quantization) | Medium-High | Understands full pipeline: fake quant in forward, STE in backward, real quant at deployment |
| Semantic Caching | Medium-High | Understood mechanism, independently connected threshold tuning to precision-recall tradeoff |
| Knowledge Distillation (mathematical) | Medium | Dark knowledge, temperature, KL divergence, combined loss with T² scaling, α tradeoff. Needs rehearsal |

---

## Practice Problems

| Problem | Result | Notes |
|---------|--------|-------|
| Quantize r=0.5 (s=0.00784, z=102) | Partial | Minor rounding error (63 vs 64), correct process |
| Dequantize q=166 back | Pass | Correctly computed 0.50176, identified quantization error |
| Cost reduction interview question | Pass | Good structure, missed batching |
| Threshold tradeoff analysis | Pass | Excellent reasoning on both directions |

---

## Key Insights Demonstrated

- Connected semantic cache threshold tuning to precision-recall tradeoff independently
- Strong practical experience with production ML systems (OOM, scaling)
- Good prioritization instinct (low-risk before high-risk optimizations)
- Correctly identified why symmetric wastes buckets for skewed distributions
- Identified tanh as good symmetric quantization candidate
- Connected temperature in distillation to InfoNCE temperature from Session 7 — strong cross-session linking
- Correctly intuited T² scaling compensation without being told the derivation

---

## Follow-up Needed

- [ ] Review STE and include in quantization interview answer
- [ ] Practice stating symmetric-for-weights, asymmetric-for-activations rule
- [ ] Add batching to cost optimization mental checklist
- [ ] Distillation interview rehearsal (didn't complete this session)
- [ ] Pruning at mathematical depth (not yet covered)
- [ ] Continue study plan: C.19 CNNs or C.20 RNNs

---

## Interview Readiness Assessment

**Topics Ready for Interview**:
- ML system cost optimization (comprehensive framework)
- Quantization math (formulas, tradeoffs, PTQ vs QAT, examples)
- QAT mechanism (fake quantization pipeline)
- Semantic caching concept and threshold tuning

**Topics Needing More Work**:
- Quantization rehearsal (include STE, usage rules, watch terminology)
- Distillation interview rehearsal (concepts understood, needs practice delivery)
- Pruning at mathematical depth (not yet covered)

**Recommended Focus for Next Session**:
- Distillation rehearsal (quick, 5 min warmup)
- Pruning math (complete the cost optimization trilogy)
- Then: C.19 CNNs or C.20 RNNs

---

## Notes for Future Sessions

- Student has strong production ML experience — leverage real-world examples
- Responds well to concrete numerical examples (quantization walkthrough was effective)
- Good at connecting concepts across domains (cache threshold → precision-recall)
- Sometimes minor arithmetic slips under pressure — practice mental math
- Prefers building up from intuition to formulas (Socratic approach works well)
