# ML Interview Study Tracker

**Last Updated**: 2026-02-17

**Target Interview Date**: [Your interview date]

**Overall Interview Readiness**: 54%

---

## Quick Stats

| Domain | Topics Covered | Total Topics | Status |
|--------|---------------|--------------|--------|
| A. ML Fundamentals (20%) | 3 | 8 | 🟡 In Progress |
| B. Classical ML (15%) | 0 | 7 | 🔴 Not Started |
| C. Deep Learning & RL (25%) | 6 (+2 in progress) | 11 | 🟡 In Progress |
| D. NLP & Multi-Modal (12%) | 1 | 10 | 🟡 In Progress |
| E. ML System Design (18%) | 8 | 8 | 🟢 Interview Ready |
| F. Practical ML (10%) | 3 | 6 | 🟡 In Progress |

**Status Legend**: 🔴 Not Started | 🟡 In Progress | 🟢 Interview Ready

---

## Study Priority (Based on Weights)

1. 🔥 **Deep Learning & RL (25%)** - Transformers, attention, CNNs, RNNs, RL, RLHF — GOOD PROGRESS
2. 🔥 **ML Fundamentals (20%)** - Gradient descent, bias-variance, regularization
3. ✅ **ML System Design (18%)** - Pipelines, serving, A/B testing — COMPLETE (8/8)
4. 📌 **Classical ML (15%)** - Trees, SVM, clustering
5. 📋 **NLP & Multi-Modal (12%)** - Embeddings, BERT, ViT, CLIP, multi-modal LLMs
6. 📋 **Practical ML (10%)** - Debugging, imbalanced data — STARTED

---

## Topics Mastered

### A. ML Fundamentals
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **A.8 Evaluation Metrics - AUC-ROC** | 2026-02-02 | Medium-High | • Measures ranking ability across all thresholds<br>• AUC = P(random positive ranks higher than random negative)<br>• Threshold-independent, handles imbalanced data<br>• AUC=0.5 (random), AUC=1.0 (perfect)<br>• Use AUC for flexible thresholds; Precision@k for fixed top-k<br>• ROC axes: x=FPR, y=TPR |
| **A.3 Gradient Descent & Optimization - Logistic Regression** | 2026-02-02 | High | • Derived complete gradient from first principles<br>• Chain rule: ∂L/∂w = (∂L/∂ŷ)(∂ŷ/∂z)(∂z/∂w)<br>• Beautiful simplification: ∂L/∂z = ŷ - y<br>• Complete gradient: ∂L/∂w = (ŷ - y)x + λw<br>• L2 regularization adds λw term (weight decay)<br>• Can derive on whiteboard for interviews |
| **A.3 Optimizers - SGD/Momentum/RMSProp/Adam/AdamW** | 2026-02-10 | Medium-High | • SGD: w = w - lr*dw (baseline)<br>• Momentum: accumulates gradient direction, cancels oscillation<br>• RMSProp: per-parameter adaptive LR via running avg of squared gradients<br>• Adam = Momentum + RMSProp + bias correction (m̂=m/(1-β₁ᵗ))<br>• β₂=0.999 needs correction longer than β₁=0.9<br>• AdamW: decouples weight decay from adaptive scaling (uniform regularization)<br>• Adam+L2 distorts regularization; AdamW applies λw directly to weights |

### B. Classical ML
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| — | — | — | — |

### C. Deep Learning & RL
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **C.21 Transformers & Self-Attention** | 2026-02-02 | High | • Self-attention solves RNN bottlenecks (parallel + direct connections)<br>• O(n²) complexity trade-off for long sequences<br>• Q, K, V mechanism: similarity-weighted information retrieval<br>• Positional encodings needed (RoPE, sinusoidal, learned)<br>• BERT (encoder, bidirectional) vs GPT (decoder, causal)<br>• Can explain architecture choices for different tasks |
| **C.21 Multi-Head Attention** | 2026-02-03 | High | • Different heads learn different relationship types (syntactic, semantic, positional)<br>• Examples: subject-verb, pronoun resolution, induction heads<br>• d_k = d_model / num_heads (e.g., 512/8 = 64)<br>• W^O projection fuses knowledge across heads<br>• More heads needed for complex sequences (more patterns to capture) |
| **C.16 Backpropagation** | 2026-02-03 | High | • Chain rule applied layer-by-layer: ∂L/∂W₁ = (ŷ-y) × W₂ × ReLU'(z₁) × x<br>• Error signal (δ) computed once per layer, reused for all gradients<br>• ∂L/∂W = δ × (input to that layer)<br>• Vanishing gradients: small weights multiply → tiny gradients<br>• Exploding gradients: large weights multiply → huge gradients<br>• Solutions: ReLU, residual connections, gradient clipping |
| **C.17 Softmax & Cross-Entropy Gradient** | 2026-02-03 | High | • Softmax: ŷᵢ = e^zᵢ / Σe^zⱼ<br>• Cross-entropy: L = -Σ yᵢ log(ŷᵢ)<br>• ∂ŷᵢ/∂zᵢ = ŷᵢ(1-ŷᵢ), ∂ŷᵢ/∂zⱼ = -ŷᵢ·ŷⱼ (i≠j)<br>• Final gradient: ∂L/∂zⱼ = ŷⱼ - yⱼ (same as binary!)<br>• One-hot vector sums to 1 → enables simplification<br>• Practical benefits: numerical stability, simple implementation |
| **C.18 BatchNorm vs LayerNorm** *(in progress)* | 2026-02-09 | Medium | • BatchNorm: across batch dim; LayerNorm: across feature dim<br>• LayerNorm for NLP: no batch dependency, same at train/test<br>• Padding pollution issue with BatchNorm on variable-length sequences<br>• RMSNorm: removes mean centering + beta, ~10-15% faster<br>• ⚠️ Review needed: BatchNorm placement (before activation), RMSNorm details |
| **C.23 Training Techniques - Unified SFT/Distillation/RL Framework** | 2026-02-10 | Medium-High | • 2x2 framework: (on/off-policy) × (sparse/dense signal)<br>• All four share gradient: weight × ∇log π_θ(y\|x)<br>• SFT weight=𝟙(y=y*), RL weight=r(x,y), Distillation weight=π_teacher<br>• On vs off-policy: who generates data (student vs fixed dataset)<br>• Sparse vs dense: one-hot/reward vs full teacher distribution<br>• IS unification: off-policy methods get π_data/π_θ correction<br>• SFT = sparse RL with indicator reward<br>• RL can surpass teacher (no ceiling); distillation bounded by teacher |
| **C.23 Knowledge Distillation (Math)** *(in progress)* | 2026-02-17 | Medium | • Dark knowledge: soft probabilities encode inter-class relationships<br>• Temperature T softens distributions: p_i = e^(z_i/T) / Σe^(z_j/T)<br>• KL divergence loss (KL=0 when distributions match)<br>• Combined loss: L = α·T²·KL(teacher‖student) + (1-α)·CE(y, student)<br>• T² scaling compensates 1/T² gradient reduction from temperature<br>• α tradeoff: α=1 pure distillation (bounded by teacher), α=0 normal training<br>• ⚠️ Needs interview rehearsal next session |

### D. NLP & Multi-Modal
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **D.31 Contrastive Learning (InfoNCE)** | 2026-02-10 | High | • InfoNCE = cross-entropy over positive + in-batch negatives<br>• Temperature τ controls softmax sharpness (small=peaky, large=smooth)<br>• Larger batch = more hard negatives = finer-grained representations<br>• Projection head buffers encoder from info loss (discard after training)<br>• More important with aggressive augmentations<br>• InfoNCE preferred over triplet loss: richer gradients, no hard-negative mining needed |

### E. ML System Design
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **E.34 End-to-End ML Pipeline Design** | 2026-02-03 | Medium-High | • Start with business metrics, not models<br>• Pipeline: Requirements → Data → Features → Model → Serving → Evaluation<br>• Baseline first (heuristics/logistic regression), iterate to complexity<br>• Always frame improvements relative to current system |
| **E.34 Recommendation Systems - ID Embeddings & Cold Start** | 2026-02-10 | High | • Simple lookup (small scale) vs hashing/compositional embeddings (large scale)<br>• Multiple hash functions + sum/concat to reduce collisions<br>• Sum is lossy but memory efficient; concat preserves info but doubles dim<br>• Cold start: side features, content similarity, two-tower architecture<br>• Feedback loop problem: no exposure → no data → lower ranking → spiral<br>• Exploration strategies: epsilon-greedy, Thompson sampling, position-based |
| **E.34 Listing Recommendation System Design** | 2026-02-10 | Medium-High | • Two-sided marketplace: user satisfaction + host fairness + platform revenue<br>• Retrieval: hard filters → BM25 + embedding ANN + collaborative filtering → RRF merge<br>• Ranking: DCN (pointwise, cross features) or Transformer (listwise, inter-item reasoning)<br>• Transformer masking: encoder (full), causal (order-dependent), prefix (independent)<br>• Labels: multi-objective weighted scoring w1*P(click) + w2*P(save) + w3*P(book)<br>• Re-ranking: diversity (MMR), freshness, host fairness, sponsored, geo/price spread |
| **E.35 Feature Engineering & Feature Stores** | 2026-02-03 | Medium-High | • Offline (Spark/Hive, batch) vs Online (Flink/Redis, streaming)<br>• Training-serving skew: same feature computed differently<br>• Solutions: log-and-wait, unified computation, feature validation<br>• Some features fundamentally different: percentiles, global aggs, joins, ranks<br>• Hybrid: slow-changing offline, fast-changing online |
| **E.39 A/B Testing & Experimentation** | 2026-02-03 | Medium-High | • ML tests harder: delayed feedback, smaller effects, feedback loops<br>• Novelty effects, position bias<br>• Filter bubble: only learn about what you show<br>• Solutions: exploration (epsilon-greedy, Thompson sampling), IPW<br>• Feature leakage: temporal availability at prediction time |
| **E.40 Monitoring & Model Degradation** | 2026-02-09 | Medium-High | • Three drift types: covariate, label, concept<br>• Label shift = same pattern, different rate; Concept drift = relationship changes<br>• 4-layer monitoring: data, model, operational, business<br>• Operational monitoring segmented by pipeline step<br>• Alert tiering: P0 (immediate), P1 (hours), P2 (daily)<br>• Prevention: scheduled retraining, online learning, human-in-the-loop |
| **E.38 Model Serving & Cost Optimization** | 2026-02-17 | Medium-High | • Model compression: distillation, quantization, pruning<br>• MoE: saves compute (8/256 experts) but NOT memory (full model loaded)<br>• Smart inference: model cascading/routing, speculative decoding<br>• Infrastructure: caching (exact + semantic), request batching, auto-scaling<br>• Semantic cache: embed queries → ANN search → threshold as precision-recall tradeoff<br>• Quantization math: s=(r_max-r_min)/(2^b-1), z=round(-r_min/s), q=round(r/s)+z<br>• Symmetric (weights, centered) vs asymmetric (activations, skewed e.g. ReLU)<br>• PTQ (fast, no retrain) vs QAT (better accuracy, uses STE for backprop through round)<br>• Cost prioritization: right-size → scaling → caching → cascading → compression |

### F. Practical ML
| Topic | Date Mastered | Confidence | Key Points |
|-------|---------------|------------|------------|
| **F.43 Handling Imbalanced Data** | 2026-02-09 | Medium-High | • Weighted cross-entropy: weight rare class more, derived gradient (49x larger for minority)<br>• Focal loss: (1-ŷ)^γ modulator, γ=0 reduces to weighted CE (RetinaNet, 2017)<br>• Sampling: oversampling/SMOTE, undersampling, data augmentation<br>• Threshold tuning as simplest first approach<br>• Connected to AUC-ROC and AUC-PR for evaluation |
| **F.46 Model Interpretability (SHAP)** | 2026-02-09 | Medium | • Shapley values from game theory: average marginal contribution across all orderings<br>• Exact computation is O(n!) — intractable<br>• Approximations: TreeSHAP O(TLD²), KernelSHAP, DeepSHAP<br>• Advantages over feature importance: local explanations, directionality, theoretical guarantees<br>• Guarantees: efficiency (sum to prediction), symmetry, null player |
| **F.42 Debugging Training Issues** | 2026-02-09 | Medium-High | • Debugging hierarchy: data → sanity checks → training mechanics → regularization<br>• Initial loss sanity check: should be log(k) for k classes; LLMs ~10.4 for 32k vocab<br>• Overfit tiny batch as first diagnostic (validates entire pipeline)<br>• NaN causes: log(0), exp overflow, 0/0 derivatives, exploding gradients<br>• Sudden NaN: weight growth, model overconfidence, bad batch, LR schedule<br>• Overfitting (train↓ eval↑) vs underfitting (both high) — opposite fixes |

---

## Knowledge Gaps

### 🔴 High Priority (Must fix before interview)
- None identified yet

### 🟡 Medium Priority (Should review)
- Full end-to-end system design practice (three practices done — delivery improving steadily)
- System design: always mention label sources early in summary
- System design: be specific with feature examples (enumerate, don't generalize)
- BatchNorm placement details (before vs after activation — original paper says before)
- RMSNorm precise mechanics (removes mean centering + beta, not variance)
- ROC axes confusion (recurring — swapped axes again in Session 5, need drilling)
- AUC-PR interpretation nuances (don't compare to 0.5, compare to positive class rate)

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
- [x] Compare AUC-ROC vs AUC-PR for imbalanced data
- [ ] Calculate precision, recall, F1 from confusion matrix

**Classical ML**
- [ ] Explain how random forests reduce variance
- [x] Derive logistic regression gradient (with L2 regularization!)
- [ ] Explain SVM margin and kernel trick
- [ ] Describe K-means algorithm and limitations

**Deep Learning & RL**
- [x] Walk through backpropagation step by step
- [x] Explain vanishing gradients and solutions
- [x] Describe attention mechanism and transformers in detail
- [x] Compare BERT vs GPT architectures and use cases
- [x] Explain multi-head attention and W^O projection
- [x] Derive softmax + cross-entropy gradient
- [ ] Compare batch norm vs layer norm (in progress — review details)
- [x] Explain unified SFT/Distillation/RL gradient framework
- [ ] Explain RL fundamentals (MDP, Bellman, on/off-policy)
- [ ] Describe policy gradient methods (REINFORCE, PPO, GRPO)
- [ ] Explain RLHF and DPO for LLM alignment

**NLP & Multi-Modal**
- [ ] Explain Word2Vec (skip-gram and CBOW)
- [x] Describe transformer architecture
- [ ] Explain BERT pre-training objectives
- [ ] Explain Vision Transformers (ViT) and patch embeddings
- [x] Explain InfoNCE loss and contrastive learning fundamentals
- [ ] Describe CLIP and text-image contrastive learning
- [ ] Explain multi-modal LLM architectures
- [ ] Describe diffusion models and video generation

**System Design**
- [x] Design an end-to-end ML pipeline
- [x] Explain A/B testing for ML models
- [x] Discuss feature store architecture and tradeoffs
- [x] Identify and prevent feature leakage
- [x] Handle data drift scenarios (covariate, label, concept drift)
- [x] Design a monitoring framework (4-layer: data, model, operational, business)
- [x] Discuss model serving trade-offs and cost optimization

---

## Study Plan

### This Week's Focus
1. [x] A.8 AUC-ROC and evaluation metrics (COMPLETED)
2. [x] A.3 Logistic regression gradient derivation with L2 regularization (COMPLETED)
3. [x] C.21 Transformers & self-attention mechanism (COMPLETED)
4. [x] Multi-head attention - why multiple heads? (COMPLETED)
5. [x] C.16 Backpropagation for simple neural network (COMPLETED)
6. [x] Softmax & cross-entropy gradient (multi-class extension) (COMPLETED)
7. [x] E.34 End-to-end ML pipeline design (COMPLETED)

### Upcoming Topics
- [x] E.40 Monitoring and model degradation (COMPLETED 2026-02-09)
- [x] C.23 Unified SFT/Distillation/RL framework (COMPLETED 2026-02-10)
- [ ] Batch norm vs Layer norm (IN PROGRESS — review details needed)
- [ ] C.19 CNNs (convolutions, pooling, architectures)
- [ ] C.20 RNNs, LSTMs, GRUs (vanishing gradients, gating)
- [ ] A.5 Bias-variance tradeoff
- [ ] B.10 Decision trees, random forests, gradient boosting

### Review Scheduled
- [ ] Multi-head attention (reinforce W^O understanding)
- [ ] Backprop derivation (practice on whiteboard)
- [ ] ROC axes drill (recurring confusion — FPR on x, TPR on y)
- [ ] SHAP practice (explain Shapley values fluently)

---

## Session History Summary

| Date | Topics Covered | Key Wins | Gaps Found |
|------|---------------|----------|------------|
| 2026-02-02 (Session 1) | A.8 AUC-ROC evaluation metric | • Understood ROC curve (corrected axis confusion)<br>• Mastered AUC vs Precision@k tradeoffs<br>• Can apply to real scenarios (rec sys, fraud detection)<br>• Interview-ready for AUC questions | • Initially confused ROC axes (resolved)<br>• Needed clarification on metric selection (resolved) |
| 2026-02-02 (Session 2) | A.3 Logistic regression gradient w/ L2 regularization | • **First use of 3-step structured workflow - success!**<br>• Derived complete gradient from first principles<br>• Mastered chain rule application in ML<br>• Understood beautiful simplification: ∂L/∂z = ŷ - y<br>• Can perform whiteboard derivation<br>• Grasped weight decay intuition | • Chain rule was fuzzy (resolved with review)<br>• Made errors on BCE derivative (corrected)<br>• Minor: didn't cover bias gradient or batch averaging |
| 2026-02-02 (Session 3) | C.21 Transformers & self-attention mechanism | • **Student had exceptional baseline knowledge!**<br>• Structured understanding into interview-ready format<br>• Self-attention: Q, K, V mechanism and O(n²) trade-off<br>• Positional encodings: RoPE, sinusoidal, learned<br>• BERT vs GPT: encoder/decoder, bidirectional/causal<br>• Applied knowledge to practical scenarios<br>• **3 topics mastered in one day!** | • Minor: didn't know about causal masking in GPT (added)<br>• Minor: less familiar with all positional encoding types (covered) |
| 2026-02-03 (Session 4) | Multi-head attention, Backprop, Softmax+CE, ML Pipelines | • **4 major topics in one session!**<br>• Strong math derivations for backprop and softmax<br>• Connected concepts across sessions<br>• ML System Design shows real-world experience<br>• Can whiteboard multi-head attention and gradients | • Minor derivative mechanics (corrected in session)<br>• Could use more system design practice |
| 2026-02-09 (Session 5) | E.36 Monitoring, C.18 Norms, F.39 Imbalance, A.8 AUC review, F.42 SHAP, F.38 Debugging | • Built 4-layer monitoring framework (interview-ready)<br>• Mastered drift types and weighted CE gradient derivation<br>• Learned focal loss, SHAP/Shapley values, AUC-PR<br>• Systematic debugging framework (4-row table)<br>• Strong cross-topic connections throughout<br>• **6 topics in one session — most productive yet!** | • ROC axes swapped again (recurring)<br>• BatchNorm placement corrected<br>• L1/Lasso distinction corrected<br>• Sudden NaN reasoning needed guidance |
| 2026-02-10 (Session 6) | C.23 Unified SFT/Distillation/RL Framework | • Built complete 2x2 framework (on/off-policy × sparse/dense) Socratically<br>• Derived unified gradient: weight × ∇log π_θ for all four methods<br>• Applied importance sampling to unify off-policy under on-policy expectation<br>• Proved SFT = sparse RL with indicator reward<br>• Strong practical trade-off reasoning (RL vs distillation) | • IS application: forgot π_data in numerator (minor)<br>• Initially confused On-Policy Distillation with RL (added reward where none exists) |
| 2026-02-10 (Session 7) | D.31 Contrastive Learning, InfoNCE, ID Embeddings, Cold Start | • InfoNCE formula & cross-entropy connection mastered<br>• Temperature, batch size, projection head practical details<br>• ID embedding pipeline: simple lookup vs compositional embeddings<br>• Cold start: user & item sides, two-tower architecture<br>• Independently identified feedback loop / popularity bias problem<br>• 8 interconnected concepts in one session | • Initially wrote "InfoBCE" (typo, corrected)<br>• Temperature framing: said "randomization" instead of "sharpness" (minor) |
| 2026-02-10 (Session 8) | E.34 System Design: Airbnb Relisting Detection | • First full end-to-end system design practice<br>• Strong problem framing (fraud/imbalanced, precision priority)<br>• Good retrieval design (geo filter + FAISS ANN)<br>• Solid feature categories and model selection philosophy<br>• Connected contrastive learning to practical system<br>• Label flywheel and tiered action concepts | • Interview delivery: jumped to features before pipeline structure<br>• Behavioral/network signals needed prompting<br>• Adversarial adaptation answer too vague initially |
| 2026-02-10 (Session 9) | E.34 System Design: Listing Recommendation | • Pipeline-first delivery (big improvement over Session 8)<br>• Multi-stakeholder framing (two-sided marketplace)<br>• Deep ranking architecture discussion (DCN vs transformer, masking patterns)<br>• Creative prefix-mask transformer proposal<br>• Connected cold start/feedback loop across sessions<br>• Multi-objective scoring and re-ranking business logic | • Forgot hard filters in summary<br>• Missing problem framing opener and eval/monitoring closer<br>• CF retrieval path and revenue-based weights needed prompting |
| 2026-02-10 (Session 10) | E.34 System Design: Family-Friendly Listings | • Best summary delivery of all 3 designs<br>• Strong integration design (ff_scorer as filter + boost + feature)<br>• Self-identified blind spot: metrics must segment by target audience<br>• Connected AUC-PR for imbalanced from Session 5<br>• Multi-modal fusion tradeoffs (separate scores vs embedding concat)<br>• Image classification for family-friendly — strong independent idea | • Label sources missing from summary<br>• Initial feature answer too brief<br>• "Family friendly" definition initially scattered |
| 2026-02-10 (Session 11) | A.3 Optimizers: SGD → Momentum → RMSProp → Adam → AdamW | • Built complete optimizer progression step-by-step<br>• Strong momentum intuition (cancel oscillation, accumulate consistent direction)<br>• Good RMSProp numerical reasoning (amplification for rare parameters)<br>• Understood Adam bias correction and β₂ > β₁ implication<br>• AdamW: decoupled weight decay concept understood | • Bias correction: couldn't derive independently<br>• AdamW wording slightly imprecise<br>• SGD sign error (minor) |
| 2026-02-17 (Session 12) | E.38 Cost Optimization, Quantization Math, QAT, Distillation Math | • Comprehensive cost optimization framework (compression, smart inference, infrastructure)<br>• Quantization formulas derived with concrete examples<br>• Symmetric vs asymmetric tradeoffs + when-to-use rules<br>• PTQ vs QAT: fake quantization mechanism, STE, deployment pipeline<br>• Distillation math: dark knowledge, temperature, KL divergence, T² scaling<br>• Semantic caching: connected threshold to precision-recall tradeoff<br>• Strong cross-session linking (temperature ↔ InfoNCE from Session 7)<br>• **ML System Design domain complete (8/8)!** | • Missed batching in interview answer<br>• Forgot STE in quantization summary<br>• QAT terminology: "fade" vs "fake"<br>• Distillation rehearsal incomplete |

---

*This tracker is your single source of truth for interview preparation progress.*
