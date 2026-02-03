# Google's Rules of Machine Learning
## Best Practices for ML Engineering — Martin Zinkevich

This document captures Google's best practices for ML engineering. Use these rules to guide students through production ML concepts and identify common pitfalls.

---

## Overview Philosophy

> "Do machine learning like the great engineer you are, not like the great machine learning expert you aren't."

**Core approach:**
1. Make sure your pipeline is solid end-to-end
2. Start with a reasonable objective
3. Add common-sense features in a simple way
4. Make sure your pipeline stays solid

**Key insight:** Most gains come from great features, not great ML algorithms. Adding complexity slows future releases.

---

## Terminology

| Term | Definition |
|------|------------|
| **Instance** | The thing about which you want to make a prediction |
| **Label** | The answer for a prediction task (from ML system or training data) |
| **Feature** | A property of an instance used in prediction |
| **Feature Column** | A set of related features (e.g., all possible countries) |
| **Example** | An instance (with features) and a label |
| **Model** | A statistical representation of a prediction task |
| **Metric** | A number you care about (may or may not be optimized) |
| **Objective** | A metric your algorithm is trying to optimize |
| **Pipeline** | Infrastructure surrounding ML: data gathering → training → serving |

---

## Before Machine Learning (Rules 1-3)

### Rule #1: Don't be afraid to launch a product without machine learning
- ML requires data; without it, heuristics often work better
- If ML gives 100% boost, heuristics get you 50% there
- Examples: rank by install rate, filter known spam publishers, rank contacts by recency
- **Don't use ML until you have data**

**Interview relevance:** Shows understanding that ML isn't always the answer

### Rule #2: Make metrics design and implementation a priority
- Track everything before building ML
- Get historical data early
- Metrics reveal what changes and what doesn't

**Interview relevance:** Data-driven decision making, instrumentation

### Rule #3: Choose machine learning over a complex heuristic
- Simple heuristic → get product out
- Complex heuristic → unmaintainable
- ML models are easier to update than complex rules

**Interview relevance:** When to use ML vs rules

---

## ML Phase I: Your First Pipeline (Rules 4-15)

### Infrastructure (Rules 4-6)

#### Rule #4: Keep the first model simple and get the infrastructure right
- First model = biggest boost, doesn't need to be fancy
- Focus on: getting data in, defining good/bad, integrating model
- Simple features ensure correct data flow

**Interview relevance:** System design, incremental development

#### Rule #5: Test the infrastructure independently from the ML
- Test data input separately
- Test model export separately
- Encapsulate learning parts for testing

**Interview relevance:** ML testing strategies

#### Rule #6: Be careful about dropped data when copying pipelines
- Old pipelines may drop data you need
- Check for filtering that doesn't apply to new use case

**Interview relevance:** Data quality, debugging pipelines

### Heuristics & Features (Rule 7)

#### Rule #7: Turn heuristics into features, or handle them externally
- **Preprocess** using heuristic (e.g., blacklist)
- **Create feature** from heuristic score
- **Mine raw inputs** of heuristic
- **Modify labels** based on heuristic

**Interview relevance:** Feature engineering from domain knowledge

### Monitoring (Rules 8-11)

#### Rule #8: Know the freshness requirements of your system
- How much does performance degrade with stale model?
- Daily? Weekly? Monthly updates needed?
- Freshness needs change as features change

**Interview relevance:** Model maintenance, SLAs

#### Rule #9: Detect problems before exporting models
- Sanity checks on held-out data before export
- Check AUC, calibration before serving
- Don't impact users with bad models

**Interview relevance:** Model validation, deployment gates

#### Rule #10: Watch for silent failures
- Stale tables can go unnoticed for months
- Track data statistics
- Manually inspect data occasionally

**Interview relevance:** Monitoring, alerting

#### Rule #11: Give feature columns owners and documentation
- Know who maintains each feature
- Document what features are and where they come from

**Interview relevance:** ML ops, team organization

### Your First Objective (Rules 12-15)

#### Rule #12: Don't overthink which objective to directly optimize
- Early on, all metrics tend to go up together
- Keep it simple initially
- Revise objective if needed later

**Interview relevance:** Objective function design

#### Rule #13: Choose a simple, observable and attributable metric for your first objective
- Was link clicked? Object downloaded? Forwarded?
- Avoid indirect effects initially (visit next day, session length)
- Use indirect effects for A/B testing decisions

**Interview relevance:** Metric selection

#### Rule #14: Starting with an interpretable model makes debugging easier
- Linear/logistic regression easier to debug
- Predictions interpretable as probabilities
- Check calibration to find issues

**Interview relevance:** Model selection tradeoffs

#### Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer
- Quality ranking = fine art
- Spam filtering = war (adversarial)
- Keep spam models updating frequently

**Interview relevance:** System architecture, adversarial ML

---

## ML Phase II: Feature Engineering (Rules 16-28)

### Launch & Iterate (Rule 16)

#### Rule #16: Plan to launch and iterate
- This won't be your last model
- Think about ease of adding/removing features
- Launch models regularly (quarterly+)

**Interview relevance:** Agile ML development

### Feature Selection (Rules 17-22)

#### Rule #17: Start with directly observed and reported features as opposed to learned features
- Avoid deep learning features initially
- External system features can become stale
- Get baseline with simple features first

**Interview relevance:** Feature engineering philosophy

#### Rule #18: Explore with features of content that generalize across contexts
- Use signals from other parts of the product
- Watch counts, co-watches, explicit ratings
- Helps with new content cold start

**Interview relevance:** Transfer learning, cold start

#### Rule #19: Use very specific features when you can
- Millions of simple features > few complex ones
- Document IDs, query IDs for head queries
- Use regularization to prune rare features

**Interview relevance:** Feature granularity tradeoffs

#### Rule #20: Combine and modify existing features to create new features in human-understandable ways
- Discretization: continuous → buckets
- Crosses: combine feature columns
- Keep transformations interpretable

**Interview relevance:** Feature crosses, binning

#### Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have
- More data → more features possible
- Regularization helps with limited data

**Interview relevance:** Model capacity, overfitting

#### Rule #22: Clean up features you are no longer using
- Remove unused features
- Reduce infrastructure complexity
- Easier maintenance

**Interview relevance:** Technical debt

### Human Analysis (Rules 23-28)

#### Rule #23: You are not a typical end user
- Don't trust your own intuition about features
- Test on real users

**Interview relevance:** Avoiding bias

#### Rule #24: Measure the delta between models
- Compare new model to current one
- Look at specific examples that changed
- Understand why predictions differ

**Interview relevance:** Model comparison, A/B testing

#### Rule #25: When choosing models, utilitarian performance trumps predictive power
- Real-world impact matters more than AUC
- Consider user experience, business metrics

**Interview relevance:** Metrics that matter

#### Rule #26: Look for patterns in the measured errors, and create new features
- Analyze failure cases
- Create features to address systematic errors

**Interview relevance:** Error analysis, iterative improvement

#### Rule #27: Try to quantify observed undesirable behavior
- If you see bad results, measure them
- Create metrics for failure modes

**Interview relevance:** Debugging, problem quantification

#### Rule #28: Be aware that identical short-term behavior does not imply identical long-term behavior
- Models can have same accuracy but different long-term effects
- Consider feedback loops, user behavior changes

**Interview relevance:** Long-term thinking, feedback loops

---

## Training-Serving Skew (Rules 29-37)

**This is one of the most critical topics for ML interviews!**

#### Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time
- Log features at serving time
- Use logged features for training
- Guarantees consistency

**Interview relevance:** Training-serving consistency (CRITICAL)

#### Rule #30: Importance weight sampled data, don't arbitrarily drop it!
- If you sample data, weight it appropriately
- Random dropping introduces bias

**Interview relevance:** Sampling strategies

#### Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change
- Tables update over time
- Training data may have different values than serving

**Interview relevance:** Data freshness, joins

#### Rule #32: Reuse code between your training pipeline and your serving pipeline whenever possible
- Same code = same behavior
- Reduces bugs and inconsistencies

**Interview relevance:** Code reuse, consistency

#### Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after
- Don't test on training data
- Use temporal splits for time-series data

**Interview relevance:** Evaluation methodology

#### Rule #34: In binary classification for filtering (such as spam detection), make small short-term sacrifices in performance for very clean data
- Clean data > slightly better model
- Especially important for adversarial problems

**Interview relevance:** Data quality tradeoffs

#### Rule #35: Beware of the inherent skew in ranking problems
- Position bias: items shown higher get more clicks
- Clicks are biased by ranking

**Interview relevance:** Position bias, ranking metrics

#### Rule #36: Avoid feedback loops with positional features
- Positional features can create self-reinforcing loops
- Be careful with features derived from model outputs

**Interview relevance:** Feedback loops

#### Rule #37: Measure Training/Serving Skew
- Difference between training and serving can kill performance
- Check feature distributions match
- Monitor for drift

**Interview relevance:** Monitoring, distribution shift (CRITICAL)

---

## ML Phase III: Slowed Growth (Rules 38-43)

### Optimization Refinement (Rules 38-40)

#### Rule #38: Don't waste time on new features if unaligned objectives have become the issue
- If metrics aren't aligned with goals, fix objectives first
- More features won't help misaligned objectives

**Interview relevance:** Objective alignment

#### Rule #39: Launch decisions will depend upon more than one metric
- Don't optimize a single number blindly
- Consider multiple stakeholders and metrics

**Interview relevance:** Multi-objective optimization

#### Rule #40: Keep ensembles simple
- Simple ensembles (averaging) often best
- Complex ensembles hard to debug

**Interview relevance:** Ensemble methods

### Complex Models (Rules 41-43)

#### Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals
- When gains slow, find new data
- Look at user relationships, external data
- Consider different problem formulations

**Interview relevance:** Breaking through plateaus

#### Rule #42: Don't expect diversity, personalization, or relevance to be as correlated with popularity as you think they are
- Popular items aren't always relevant
- Diversity and personalization require explicit modeling

**Interview relevance:** Beyond popularity

#### Rule #43: Your friends tend to be the same across different products. Your interests tend not to be
- Social graph transfers well
- Interest graphs are context-dependent

**Interview relevance:** Transfer learning, social vs interest graphs

---

## Key Principles Summary

1. **Infrastructure first, fancy ML later**
2. **Simple models + great features > complex models + simple features**
3. **Monitor everything, trust nothing**
4. **Iterate quickly, launch frequently**
5. **Training-serving skew is a silent killer**
6. **Human heuristics contain valuable signal**
7. **Know your freshness requirements**
8. **Interpretability aids debugging**

---

## Topic Mapping for Teaching

| Student Topic | Relevant Rules |
|---------------|----------------|
| Getting started with ML | 1-3 |
| Building first model | 4-7, 12-14 |
| Feature engineering | 16-22, 26 |
| Model evaluation | 9, 24-25, 33 |
| Training-serving skew | 29-37 (especially 29, 32, 37) |
| Monitoring & ops | 8-11, 37 |
| Debugging models | 14, 23-28 |
| Ranking systems | 15, 35-36 |
| Scaling/plateaus | 38-43 |
| Adversarial problems | 15, 34 |
