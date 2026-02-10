# ML Interview Study Tracker

**Last Updated**: 2026-02-09

**Target Interview Date**: [Your interview date]

**Overall Interview Readiness**: 27%

---

## Quick Stats

| Domain | Topics Covered | Total Topics | Status |
|--------|---------------|--------------|--------|
| A. ML Fundamentals (20%) | 2 | 8 | 🟡 In Progress |
| B. Classical ML (15%) | 0 | 7 | 🔴 Not Started |
| C. Deep Learning (25%) | 4 (+1 in progress) | 8 | 🟡 In Progress |
| D. NLP (12%) | 0 | 6 | 🔴 Not Started |
| E. ML System Design (18%) | 4 | 8 | 🟡 In Progress |
| F. Practical ML (10%) | 0 | 6 | 🔴 Not Started |

**Status Legend**: 🔴 Not Started | 🟡 In Progress | 🟢 Interview Ready

---

## Study Priority (Based on Weights)

1. 🔥 **Deep Learning (25%)** - Transformers, attention, CNNs, RNNs — GOOD PROGRESS
2. 🔥 **ML Fundamentals (20%)** - Gradient descent, bias-variance, regularization
3. 📌 **ML System Design (18%)** - Pipelines, serving, A/B testing — STARTED
4. 📌 **Classical ML (15%)** - Trees, SVM, clustering
5. 📋 **NLP (12%)** - Embeddings, BERT, transformers for NLP
6. 📋 **Practical ML (10%)** - Debugging, imbalanced data

---

## Topics Mastered

### A. ML Fundamentals
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **A.8 Evaluation Metrics - AUC-ROC** | 2026-02-02 | Medium-High | • Measures ranking ability across all thresholds<br>• AUC = P(random positive ranks higher than random negative)<br>• Threshold-independent, handles imbalanced data<br>• AUC=0.5 (random), AUC=1.0 (perfect)<br>• Use AUC for flexible thresholds; Precision@k for fixed top-k<br>• ROC axes: x=FPR, y=TPR |
| **A.3 Gradient Descent & Optimization - Logistic Regression** | 2026-02-02 | High | • Derived complete gradient from first principles<br>• Chain rule: ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)<br>• Beautiful simplification: ∂L/∂z = ŷ - y<br>• Complete gradient: ∂L/∂w = (ŷ - y)x + λw<br>• L2 regularization adds λw term (weight decay)<br>• Can derive on whiteboard for interviews |

### B. Classical ML
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| — | — | — | — |

### C. Deep Learning
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **C.21 Transformers & Self-Attention** | 2026-02-02 | High | • Self-attention solves RNN bottlenecks (parallel + direct connections)<br>• O(n²) complexity trade-off for long sequences<br>• Q, K, V mechanism: similarity-weighted information retrieval<br>• Positional encodings needed (RoPE, sinusoidal, learned)<br>• BERT (encoder, bidirectional) vs GPT (decoder, causal)<br>• Can explain architecture choices for different tasks |
| **C.21 Multi-Head Attention** | 2026-02-03 | High | • Different heads learn different relationship types (syntactic, semantic, positional)<br>• Examples: subject-verb, pronoun resolution, induction heads<br>• d_k = d_model / num_heads (e.g., 512/8 = 64)<br>• W^O projection fuses knowledge across heads<br>• More heads needed for complex sequences (more patterns to capture) |
| **C.16 Backpropagation** | 2026-02-03 | High | • Chain rule applied layer-by-layer: ∂L/∂W₁ = (ŷ-y) × W₂ × ReLU'(z₁) × x<br>• Error signal (δ) computed once per layer, reused for all gradients<br>• ∂L/∂W = δ × (input to that layer)<br>• Vanishing gradients: small weights multiply → tiny gradients<br>• Exploding gradients: large weights multiply → huge gradients<br>• Solutions: ReLU, residual connections, gradient clipping |
| **C.17 Softmax & Cross-Entropy Gradient** | 2026-02-03 | High | • Softmax: ŷᵢ = e^zᵢ / Σe^zⱼ<br>• Cross-entropy: L = -Σ yᵢ log(ŷᵢ)<br>• ∂ŷᵢ/∂zᵢ = ŷᵢ(1-ŷᵢ), ∂ŷᵢ/∂zⱼ = -ŷᵢ·ŷⱼ (i≠j)<br>• Final gradient: ∂L/∂zⱼ = ŷⱼ - yⱼ (same as binary!)<br>• One-hot vector sums to 1 → enables simplification<br>• Practical benefits: numerical stability, simple implementation |
| **C.18 BatchNorm vs LayerNorm** *(in progress)* | 2026-02-09 | Medium | • BatchNorm: across batch dim; LayerNorm: across feature dim<br>• LayerNorm for NLP: no batch dependency, same at train/test<br>• Padding pollution issue with BatchNorm on variable-length sequences<br>• RMSNorm: removes mean centering + beta, ~10-15% faster<br>• ⚠️ Review needed: BatchNorm placement (before activation), RMSNorm details |

### D. NLP
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| — | — | — | — |

### E. ML System Design
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **E.30 End-to-End ML Pipeline Design** | 2026-02-03 | Medium-High | • Start with business metrics, not models<br>• Pipeline: Requirements → Data → Features → Model → Serving → Evaluation<br>• Baseline first (heuristics/logistic regression), iterate to complexity<br>• Always frame improvements relative to current system |
| **E.31 Feature Engineering & Feature Stores** | 2026-02-03 | Medium-High | • Offline (Spark/Hive, batch) vs Online (Flink/Redis, streaming)<br>• Training-serving skew: same feature computed differently<br>• Solutions: log-and-wait, unified computation, feature validation<br>• Some features fundamentally different: percentiles, global aggs, joins, ranks<br>• Hybrid: slow-changing offline, fast-changing online |
| **E.35 A/B Testing & Experimentation** | 2026-02-03 | Medium-High | • ML tests harder: delayed feedback, smaller effects, feedback loops<br>• Novelty effects, position bias<br>• Filter bubble: only learn about what you show<br>• Solutions: exploration (epsilon-greedy, Thompson sampling), IPW<br>• Feature leakage: temporal availability at prediction time |
| **E.36 Monitoring & Model Degradation** | 2026-02-09 | Medium-High | • Three drift types: covariate, label, concept<br>• Label shift = same pattern, different rate; Concept drift = relationship changes<br>• 4-layer monitoring: data, model, operational, business<br>• Operational monitoring segmented by pipeline step<br>• Alert tiering: P0 (immediate), P1 (hours), P2 (daily)<br>• Prevention: scheduled retraining, online learning, human-in-the-loop |

### F. Practical ML
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| — | — | — | — |

---

## Knowledge Gaps

### 🔴 High Priority (Must fix before interview)
- None identified yet

### 🟡 Medium Priority (Should review)
- Full end-to-end system design practice (need structured practice)
- BatchNorm placement details (before vs after activation — original paper says before)
- RMSNorm precise mechanics (removes mean centering + beta, not variance)

### 🟢 Recently Resolved
| Gap | Resolution Date | Notes |
|-----|-----------------|-------|
| Chain rule application in multi-layer networks | 2026-02-03 | Covered in backprop derivation |
| Softmax derivative mechanics | 2026-02-03 | Derived both cases (i=j, i≠j) |
| Monitoring & model degradation in production | 2026-02-09 | Built 4-layer monitoring framework, mastered drift types |

---

## Interview Readiness Checklist

### Can You Confidently...

**Fundamentals**
- [x] Derive gradient descent update rules
- [ ] Explain bias-variance tradeoff with examples
- [x] Compare L1 vs L2 regularization (L2 covered in logistic regression)
- [x] Explain AUC-ROC and when to use vs Precision@k
- [ ] Calculate precision, recall, F1 from confusion matrix

**Classical ML**
- [ ] Explain how random forests reduce variance
- [x] Derive logistic regression gradient (with L2 regularization!)
- [ ] Explain SVM margin and kernel trick
- [ ] Describe K-means algorithm and limitations

**Deep Learning**
- [x] Walk through backpropagation step by step
- [x] Explain vanishing gradients and solutions
- [x] Describe attention mechanism and transformers in detail
- [x] Compare BERT vs GPT architectures and use cases
- [x] Explain multi-head attention and W^O projection
- [x] Derive softmax + cross-entropy gradient
- [ ] Compare batch norm vs layer norm (in progress — review details)

**NLP**
- [ ] Explain Word2Vec (skip-gram and CBOW)
- [x] Describe transformer architecture
- [ ] Explain BERT pre-training objectives

**System Design**
- [x] Design an end-to-end ML pipeline
- [x] Explain A/B testing for ML models
- [x] Discuss feature store architecture and tradeoffs
- [x] Identify and prevent feature leakage
- [x] Handle data drift scenarios (covariate, label, concept drift)
- [x] Design a monitoring framework (4-layer: data, model, operational, business)
- [ ] Discuss model serving trade-offs

---

## Study Plan

### This Week's Focus
1. [x] A.8 AUC-ROC and evaluation metrics (COMPLETED)
2. [x] A.3 Logistic regression gradient derivation with L2 regularization (COMPLETED)
3. [x] C.21 Transformers & self-attention mechanism (COMPLETED)
4. [x] Multi-head attention - why multiple heads? (COMPLETED)
5. [x] C.16 Backpropagation for simple neural network (COMPLETED)
6. [x] Softmax & cross-entropy gradient (multi-class extension) (COMPLETED)
7. [x] E.30 End-to-end ML pipeline design (COMPLETED)

### Upcoming Topics
- [x] E.36 Monitoring and model degradation (COMPLETED 2026-02-09)
- [ ] Batch norm vs Layer norm (IN PROGRESS — review details needed)
- [ ] C.19 CNNs (convolutions, pooling, architectures)
- [ ] C.20 RNNs, LSTMs, GRUs (vanishing gradients, gating)
- [ ] A.5 Bias-variance tradeoff
- [ ] B.10 Decision trees, random forests, gradient boosting

### Review Scheduled
- [ ] Multi-head attention (reinforce W^O understanding)
- [ ] Backprop derivation (practice on whiteboard)

---

## Session History Summary

| Date | Topics Covered | Key Wins | Gaps Found |
|------|---------------|----------|------------|
| 2026-02-02 (Session 1) | A.8 AUC-ROC evaluation metric | • Understood ROC curve (corrected axis confusion)<br>• Mastered AUC vs Precision@k tradeoffs<br>• Can apply to real scenarios (rec sys, fraud detection)<br>• Interview-ready for AUC questions | • Initially confused ROC axes (resolved)<br>• Needed clarification on metric selection (resolved) |
| 2026-02-02 (Session 2) | A.3 Logistic regression gradient w/ L2 regularization | • **First use of 3-step structured workflow - success!**<br>• Derived complete gradient from first principles<br>• Mastered chain rule application in ML<br>• Understood beautiful simplification: ∂L/∂z = ŷ - y<br>• Can perform whiteboard derivation<br>• Grasped weight decay intuition | • Chain rule was fuzzy (resolved with review)<br>• Made errors on BCE derivative (corrected)<br>• Minor: didn't cover bias gradient or batch averaging |
| 2026-02-02 (Session 3) | C.21 Transformers & self-attention mechanism | • **Student had exceptional baseline knowledge!**<br>• Structured understanding into interview-ready format<br>• Self-attention: Q, K, V mechanism and O(n²) trade-off<br>• Positional encodings: RoPE, sinusoidal, learned<br>• BERT vs GPT: encoder/decoder, bidirectional/causal<br>• Applied knowledge to practical scenarios<br>• **3 topics mastered in one day!** | • Minor: didn't know about causal masking in GPT (added)<br>• Minor: less familiar with all positional encoding types (covered) |
| 2026-02-03 (Session 4) | Multi-head attention, Backprop, Softmax+CE, ML Pipelines | • **4 major topics in one session!**<br>• Strong math derivations for backprop and softmax<br>• Connected concepts across sessions<br>• ML System Design shows real-world experience<br>• Can whiteboard multi-head attention and gradients | • Minor derivative mechanics (corrected in session)<br>• Could use more system design practice |
| 2026-02-09 (Session 5) | E.36 Monitoring & Degradation, C.18 BatchNorm vs LayerNorm | • Built 4-layer monitoring framework (interview-ready)<br>• Mastered drift types (covariate, label, concept)<br>• Strong operational monitoring with segmentation<br>• Understood BatchNorm vs LayerNorm trade-offs<br>• Quiz: weighted cross-entropy derivation started | • Label shift vs concept drift initially unclear (resolved)<br>• BatchNorm placement (corrected: before activation)<br>• RMSNorm details slightly inaccurate (corrected) |

---

*This tracker is your single source of truth for interview preparation progress.*
