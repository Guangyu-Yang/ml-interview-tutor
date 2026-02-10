# Session 6 Notes - 2026-02-10

## Session Overview
- **Date**: 2026-02-10
- **Main Topic**: Unified Framework for SFT, Distillation, and RL (from Zhihu article)
- **Category**: C.23 Training Techniques / D.29 Fine-tuning and Transfer Learning
- **Depth**: Mathematical Deep Dive
- **Format**: Socratic walkthrough of 2x2 framework (on/off-policy x sparse/dense signal)

## Source Material
- Zhihu article discussing the gradient-level unification of SFT, Off-Policy Distillation, RL, and On-Policy Distillation
- References: Self-Distilled Reasoner, Self-Distillation Enables Continual Learning, RL via Self-Distillation, On the Generalization of SFT

## Student's Initial Understanding
- **On/off-policy**: Had a roughly correct but slightly imprecise understanding. Described off-policy as "deploy model online, collect data, train separate model." Refined to the simpler core: who generated the data — the current model (on-policy) or something else (off-policy)?
- **SFT loss**: Solid understanding of NLL / cross-entropy over tokens. Correctly identified maximizing likelihood of next token.

## Concepts Covered

### 1. Sparse vs Dense Signal
- **Student quickly grasped**: one-hot (sparse) = only the ground truth token gets signal; teacher distribution (dense) = full vocabulary gets signal
- **Key insight student articulated**: "teacher model gives the full output distribution... student can learn how the teacher model made the tradeoff among tokens"

### 2. The 2x2 Framework

|  | Sparse Signal | Dense Signal |
|--|--|--|
| Off-Policy | SFT (one-hot, human data) | Off-Policy Distillation (teacher dist, fixed data) |
| On-Policy | RL (reward/advantage) | On-Policy Distillation (teacher dist, student data) |

- Student correctly identified all four quadrants through guided questioning
- Correctly stated Off-Policy Distillation loss is KL divergence between teacher and student
- Under Forward KL, this simplifies to cross-entropy (same machinery as SFT, just soft labels)

### 3. Unified Gradient Form: weight x nabla log pi_theta
- All four methods share: `weight × ∇log π_θ(y|x)`
- Student correctly identified weights for SFT (indicator), RL (reward), Off-Policy Distillation (π_teacher)
- **Error**: Initially guessed On-Policy Distillation weight = `reward × π_teacher` — corrected that distillation doesn't involve reward. Weight is just `π_teacher`, with the on-policy part being about where y is sampled from.

### 4. Importance Sampling Unification
- Student knew IS formula: `E_{y~p}[f(y)] = E_{y~q}[p(y)/q(y) × f(y)]`
- Applied IS to SFT gradient — **minor error**: initially wrote `𝟙(y=y*)/π_θ` instead of `(π_data/π_θ) × 𝟙(y=y*)`, forgetting π_data in numerator. Corrected.
- Applied IS to Off-Policy Distillation — correct content but **typo**: wrote E_{y~π_data} instead of E_{y~π_θ} on the result side.
- Successfully identified both patterns in unified table:
  1. Off vs on-policy: presence/absence of IS ratio π_data/π_θ
  2. Sparse vs dense: 𝟙/reward vs π_teacher

### 5. SFT as Sparse RL
- Student's intuition: "In SFT, there's an implication that the reward is 1, because we choose and trust this data"
- Formalized: SFT = RL where r(x,y) = (π_data/π_θ) × 𝟙(y=y*)
- Student understood this deeply

### 6. Practical Trade-offs (Interview Question)
- **RL vs On-Policy Distillation**: Student gave strong answer:
  - RL when rewards are verifiable (code, math) — no teacher needed
  - Distillation when rewards are hard to define but teacher is available
  - Practical cost: teacher models expensive, may not expose logits (closed-source APIs)
- **Added insight**: RL can surpass the teacher (no ceiling); distillation is bounded by teacher quality. DeepSeek-R1 example.

## Comprehension Assessment

| Concept | Understanding Level | Notes |
|---------|-------------------|-------|
| On-policy vs off-policy distinction | High | Initially verbose, refined to clean definition |
| Sparse vs dense signal | High | Grasped immediately, good intuition |
| 2x2 framework | High | Can reproduce and explain all four quadrants |
| Unified gradient form | Medium-High | Got 3/4 weights right first try, corrected 4th |
| Importance sampling trick | Medium-High | Knows formula, made minor errors in application |
| SFT = sparse RL | High | Good intuitive and formal understanding |
| Practical trade-offs | High | Strong practical reasoning about when to use each |

## Knowledge Gaps Identified
- Importance sampling application: needs more practice with mechanical steps (forgetting terms in numerator)
- On-Policy Distillation: initially confused with RL (adding reward where there is none) — resolved

## Key Wins
- Built complete 2x2 mental model from scratch through guided questioning
- Can explain the framework at both intuitive and mathematical levels
- Strong practical reasoning about trade-offs (interview-ready)
- Connected to real systems (DeepSeek-R1, closed-source API limitations)

## Follow-up Topics
- GRPO details: group-relative advantage computation
- Reverse KL vs Forward KL in distillation
- DPO / RLHF as additional training paradigms
- Practical implementation: how to do on-policy distillation efficiently
