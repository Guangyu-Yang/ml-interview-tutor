# Session Notes - February 2, 2026

## Session Overview
- **Date**: 2026-02-02
- **Duration**: ~20 minutes
- **Main Topic**: AUC-ROC (Area Under the Curve) evaluation metric
- **Format**: Quiz-based learning with Socratic method

---

## Student's Initial Understanding

**What they knew:**
- Uses AUC to evaluate recommendation system models
- Knew AUC involves a curve with TPR and FPR
- Understood it's for classification evaluation

**Misconceptions identified:**
- ❌ Had ROC axes swapped (said x-axis is TPR, y-axis is FPR - it's reversed)
- ❌ Vague understanding of "classification correctness problem"
- ❌ Didn't understand how AUC handles imbalanced data
- ❌ Unclear on AUC vs Precision@k tradeoffs

---

## Teaching Approach

### 1. Axis Correction (First Misconception)
- Immediately corrected the axis confusion
- Clarified: x-axis = FPR, y-axis = TPR
- Emphasized why this matters for interpretation

### 2. Imbalanced Data Example
Used concrete scenario:
- 95% negative examples, 5% positive examples
- Model A: Always predicts negative → 95% accuracy
- Model B: Learns patterns → 85% accuracy but finds positives
- Guided student to understand why Model B is better

### 3. Core AUC Explanation
Explained that AUC measures:
- **Ranking ability**: Can the model rank positives higher than negatives?
- **Threshold independence**: Works across all thresholds
- **Probabilistic interpretation**: P(random positive ranks higher than random negative)

### 4. AUC Values
- AUC = 0.5 → Random guessing (diagonal line)
- AUC > 0.5 → Better than random
- AUC < 0.5 → Worse than random (flip predictions!)
- AUC = 1.0 → Perfect separation

### 5. AUC vs Precision@k Tradeoff
**Critical interview concept:**
- Presented scenario: Model A (AUC=0.88, P@10=0.45) vs Model B (AUC=0.85, P@10=0.60)
- Student correctly chose Model B for rec sys
- Reasoning: Top-k matters more than global ranking for rec systems

### 6. When AUC Matters More
Tested with fraud detection example:
- Initially incorrect: Student thought higher AUC always better for "serious" problems
- Corrected: If investigating top 100, it's STILL a top-k problem
- Clarified when AUC is more important:
  - No fixed k
  - Multiple thresholds
  - Need flexibility in threshold setting

---

## Questions Asked to Student

1. "What do you already know about AUC?" (baseline check)
2. "Which model is better? Why?" (imbalanced data scenario)
3. "How would AUC help you distinguish between these models?"
4. "What does AUC = 0.5 tell you?" (✅ Correct: random guessing)
5. "Why is AUC useful for rec sys?" (✅ Correct: imbalanced data)
6. "What does AUC = 1.0 mean?" (✅ Correct: perfect prediction)
7. "Model A vs Model B - which to deploy?" (✅ Correct: chose Model B with P@10=0.60)
8. "Fraud detection scenario - still pick lower AUC?" (❌ Initially incorrect, then corrected)
9. "When would you prioritize AUC over Precision@k?" (✅ Correct after clarification)

---

## Student's Final Understanding

**Mastered concepts:**
✅ ROC curve axes (x=FPR, y=TPR)
✅ What AUC measures (ranking ability across all thresholds)
✅ Why AUC handles imbalanced data better than accuracy
✅ Interpretation of AUC values (0.5 = random, 1.0 = perfect)
✅ AUC vs Precision@k tradeoffs
✅ When to prioritize each metric

**Key insights demonstrated:**
- "Model B is better because Model A always predicts negative class"
- "AUC is not sensitive to imbalanced data"
- "If it's a top-k problem, Precision@k matters most"
- "If we don't lock to one specific k, we have flexible choice of thresholds" → use AUC

**Interview readiness for AUC:** HIGH
- Can explain what it measures
- Understands practical tradeoffs
- Knows when to use it vs alternatives

---

## Knowledge Gaps Identified

### Resolved During Session ✅
1. ~~ROC curve axes confusion~~ (corrected immediately)
2. ~~Why AUC is better than accuracy for imbalanced data~~ (explained with example)
3. ~~AUC vs Precision@k tradeoffs~~ (explored with multiple scenarios)
4. ~~When to prioritize each metric~~ (clarified with fraud detection example)

### Remaining Gaps (None for AUC)
- Topic is now interview-ready

---

## Practice Problems Worked Through

**Scenario 1: Imbalanced Data**
- 95% negative, 5% positive examples
- Compared always-negative model vs learning model
- Outcome: Understood why accuracy fails, AUC succeeds

**Scenario 2: Rec System Model Selection**
- Model A: AUC=0.88, P@10=0.45
- Model B: AUC=0.85, P@10=0.60
- Outcome: Correctly chose Model B, understood top-k importance

**Scenario 3: Fraud Detection**
- Model X: AUC=0.92, P@100=0.40
- Model Y: AUC=0.88, P@100=0.55
- Outcome: Initially incorrect, then understood it's still top-k problem

---

## Follow-Up Topics Recommended

### Immediate next topics (related to evaluation):
1. **Precision-Recall curve and PR-AUC** - Alternative to ROC for extreme imbalance
2. **Calibration** - When predictions need to be interpretable probabilities
3. **Other ranking metrics** - NDCG, MAP for rec sys

### Broader ML fundamentals to cover:
4. **Cross-validation strategies** - How to evaluate properly
5. **Confusion matrix deep dive** - Derive all metrics from scratch
6. **Statistical testing for A/B tests** - When is improvement significant?

---

## Teaching Notes

**What worked well:**
- Concrete examples (95/5 split) made abstract concept tangible
- Socratic questioning helped student discover answers
- Multiple scenarios (rec sys, fraud detection) reinforced learning
- Immediate correction of misconceptions prevented confusion

**Student learning style observed:**
- Learns well from examples
- Appreciates step-by-step explanations
- Willing to admit confusion and ask for help
- Connects concepts to practical applications (rec sys work)

**Confidence level assessment:**
- Started: Low-Medium (knew basics, had misconceptions)
- Ended: Medium-High (can explain concept and apply to scenarios)
- Interview ready: YES for AUC questions

---

## Session Success Metrics

✅ Student can explain AUC in their own words
✅ Student understands when to use AUC vs alternatives
✅ Student can answer interview-style scenario questions
✅ All initial misconceptions corrected
✅ Topic is now interview-ready

**Overall session rating: Excellent progress**
