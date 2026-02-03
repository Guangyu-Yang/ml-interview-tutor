# Session Notes: 2026-02-03

## Session Overview
- **Date**: 2026-02-03
- **Topics Covered**: Multi-head attention, Backpropagation, Softmax & Cross-entropy gradient, ML System Design (End-to-end pipelines)
- **Session Type**: Continuation of study plan + new domain exploration

---

## Topics Covered

### 1. Multi-head Attention (Deep Learning)

**Student's Initial Understanding:**
- Good intuition about CNN channel analogy (different heads = different patterns)
- Correct implementation knowledge (split d_model across heads, concatenate)
- Minor misconception: thought single-head causes tokens to focus on themselves

**Concepts Taught:**
- Different heads learn different relationship types (syntactic, semantic, positional, coreference)
- Examples: subject-verb attention, pronoun resolution, modifier detection, induction heads
- W^O projection purpose: fuses knowledge across heads, lets them "communicate"
- d_k calculation: d_model / num_heads

**Comprehension Demonstrated:**
- Correctly explained why concatenation alone isn't enough
- Articulated that heads need to "fuse their learnings" via W^O
- Understood why more heads needed for complex sequences

**Confidence Level:** High

---

### 2. Backpropagation (Deep Learning)

**Student's Initial Understanding:**
- Good high-level understanding of forward pass and error signal
- Familiar with chain rule from logistic regression derivation
- No experience manually computing gradients for MLP

**Concepts Taught:**
- Walked through 2-layer MLP: x → ReLU(W₁x + b₁) → W₂h + b₂ → MSE loss
- Derived ∂L/∂W₂ = (ŷ - y) × h
- Derived ∂L/∂W₁ = (ŷ - y) × W₂ × ReLU'(z₁) × x
- Error signal (delta) concept and computational efficiency
- Vanishing/exploding gradients explanation

**Errors Made & Corrected:**
- Initially wrote ∂L/∂ŷ = -(ŷ - y) — sign error corrected
- Initially wrote ∂ŷ/∂h = h instead of W₂ — corrected
- Confused ReLU derivative depending on z₁ vs x — corrected

**Comprehension Demonstrated:**
- Successfully computed ∂L/∂z₁ using chain rule
- Understood why small W₂ causes vanishing gradients
- Connected backprop to logistic regression derivation from previous session

**Confidence Level:** High

---

### 3. Softmax & Cross-Entropy Gradient (Deep Learning)

**Student's Initial Understanding:**
- Knew softmax formula: ŷᵢ = e^zᵢ / Σe^zⱼ
- Almost knew cross-entropy (missing negative sign)
- Correctly intuited final gradient would be ŷ - y

**Concepts Taught:**
- Derived ∂ŷᵢ/∂zᵢ = ŷᵢ(1 - ŷᵢ) using quotient rule
- Derived ∂ŷᵢ/∂zⱼ = -ŷᵢ·ŷⱼ for i ≠ j
- Combined with ∂L/∂ŷᵢ = -yᵢ/ŷᵢ
- Final result: ∂L/∂zⱼ = ŷⱼ - yⱼ
- Practical benefits: numerical stability, simple implementation

**Errors Made & Corrected:**
- Initially tried to treat denominator as constant in softmax derivative — corrected
- Confused Σyᵢ (scalar = 1) with a vector — corrected

**Comprehension Demonstrated:**
- Successfully applied quotient rule for both cases
- Understood that one-hot vector sums to 1
- Could explain why clean gradient form is practically useful

**Confidence Level:** High

---

### 4. ML System Design: End-to-End Pipelines

**Student's Initial Understanding:**
- Has real-world experience owning ML pipelines at work
- Strong instinct to start with metrics/business value, not models
- Knew key components: feature store, model store, training/eval/A/B pipelines

**Concepts Taught:**

**Feature Stores:**
- Offline (batch/Spark/Hive) vs Online (streaming/Flink/Redis)
- Training-serving skew problem
- Solutions: log-and-wait, unified computation, feature validation
- Features fundamentally different in batch vs streaming (percentiles, global aggs, joins, sessions, ranks)

**Offline Evaluation:**
- Compare to baseline, not just absolute thresholds
- Slice analysis across segments
- Error analysis and sanity checks
- Feature leakage: temporal vs sample leakage distinction

**A/B Testing for ML:**
- Differences from UI tests: delayed feedback, smaller effects, feedback loops
- Novelty effects, position bias
- Filter bubble problem
- Solutions: exploration (epsilon-greedy, Thompson sampling), inverse propensity weighting

**Comprehension Demonstrated:**
- Gave practical examples of offline vs online feature stores
- Understood training-serving skew concept
- Identified feedback loop risks
- Proposed multi-task learning with meaningful signals (conversion, likes) over clicks

**Confidence Level:** Medium-High

---

## Key Wins
- **4 major topics covered in one session** — excellent pace
- Strong mathematical derivations for backprop and softmax gradients
- Successfully connected new concepts to previous session (logistic regression gradient)
- ML System Design shows real-world experience and good instincts
- Can now explain multi-head attention, backprop, and softmax gradient on whiteboard

## Knowledge Gaps Identified
- Minor: Initially confused about some derivative mechanics (corrected during session)
- Could use more practice with full system design interviews (end-to-end scenario)
- Monitoring & model degradation not yet covered

## Follow-up Topics Needed
- Monitoring and detecting model degradation in production
- Full ML system design practice question
- Batch norm vs Layer norm (connects to transformer knowledge)
- CNNs and RNNs/LSTMs

## Interview Readiness Assessment
- **Deep Learning**: Strong — can derive key gradients, explain architectures
- **ML Fundamentals**: Strong — solid mathematical foundation
- **ML System Design**: Medium-High — good instincts, needs more structured practice

---

*Session completed successfully. Student demonstrated strong learning velocity.*
