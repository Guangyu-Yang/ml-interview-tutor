# Airbnb AI/ML Engineering Blog Posts Index

This directory contains summaries of Airbnb's AI and Machine Learning engineering blog posts. Use these to suggest relevant study topics and interview questions.

## How to Use This Index

When suggesting topics:
1. Match the student's current progress to relevant posts
2. Extract key ML concepts and techniques from relevant posts
3. Generate interview-style questions based on real Airbnb implementations
4. Prioritize posts based on interview relevance (Search/Ranking > Embeddings > NLP > CV)

---

## Posts by Category

### 🔍 Search & Ranking (HIGH PRIORITY)

#### 1. Embedding-Based Retrieval for Airbnb Search (2025)
**URL:** https://medium.com/airbnb-engineering/embedding-based-retrieval-for-airbnb-search-aabebfc85839
**Key Concepts:**
- Two-tower neural network architecture
- Contrastive learning with positive/negative pairs
- Training data from user booking journeys
- ANN indexing: IVF vs HNSW tradeoffs
- Euclidean distance vs dot product for balanced clusters
- Offline listing tower, online query tower

**Interview Topics:**
- Two-stage retrieval (candidate generation → ranking)
- How to construct training data for retrieval models
- ANN algorithm selection (IVF vs HNSW)
- Handling real-time updates in embedding indexes
- Why Euclidean distance produces more balanced clusters than dot product

**Sample Questions:**
- "Design an embedding-based retrieval system for search. What are the key components?"
- "How would you handle frequently changing inventory in an ANN index?"
- "Explain the tradeoffs between IVF and HNSW for approximate nearest neighbor search"

---

#### 2. Applying Deep Learning to Airbnb Search (2018)
**URL:** https://medium.com/airbnb-engineering/applying-deep-learning-to-airbnb-search-7ebd7230891f
**Key Concepts:**
- Evolution from GBDT to neural networks
- Challenges: scale, uniqueness of listings, sparse data
- Long-tail distribution of impressions per listing
- Need for broad generalization from few examples

**Interview Topics:**
- When to use deep learning vs traditional ML
- Handling sparse data in ranking
- Personalization with limited user history

**Sample Questions:**
- "When would you choose neural networks over gradient boosting for ranking?"
- "How do you handle the cold start problem for new listings?"

---

#### 3. Machine Learning-Powered Search Ranking of Airbnb Experiences (2019)
**URL:** https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789
**Key Concepts:**
- Stage-by-stage ML model evolution (baseline → personalization → online scoring)
- GBDT with log-loss for binary classification
- Personalization features: category intensity, recency, time-of-day fit
- Query features enabled by online scoring
- Promoting quality through weighted training labels
- Diversity in top results

**Interview Topics:**
- How to evolve ML systems as data grows
- Feature engineering for personalization
- Multi-objective optimization (bookings + quality + diversity)
- Offline vs online scoring tradeoffs

**Sample Questions:**
- "How would you add personalization to a ranking model?"
- "Explain how to balance relevance with diversity in search results"
- "How would you promote high-quality items in ranking?"

---

### 🧮 Embeddings (HIGH PRIORITY)

#### 4. Listing Embeddings for Similar Listing Recommendations (2018)
**URL:** https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
**Key Concepts:**
- Skip-gram on click sessions (treat sessions as sentences, listings as words)
- 32-dimensional embeddings
- Modifications: booked listing as global context, market negatives
- Cold-start embeddings via k-nearest neighbors
- Real-time personalization: EmbClickSim, EmbSkipSim features

**Interview Topics:**
- Word2Vec adaptation for non-text domains
- Negative sampling strategies
- Cold start solutions for embeddings
- Using embeddings for real-time personalization

**Sample Questions:**
- "How would you create embeddings for listings? What's the training signal?"
- "Explain the role of negative sampling in embedding training"
- "How do you handle embeddings for new items with no history?"

---

### 🤖 NLP & Conversational AI

#### 5. Task-Oriented Conversational AI in Customer Support (2021)
**URL:** https://medium.com/airbnb-engineering/task-oriented-conversational-ai-in-airbnb-customer-support-5ebf49169eaa
**Key Concepts:**
- Multi-layer intent detection (domain → specific intent)
- Q&A model for intent understanding (single-choice binary classification)
- Pre-training: in-domain MLM + cross-domain task finetuning
- Multilingual model with translated training data (XLM-RoBERTa)
- Multi-turn dialog state tracking
- GPU serving for transformer models (3x faster)
- Contextual bandit for online optimization

**Interview Topics:**
- Task-oriented dialog system design
- Intent classification approaches
- Transfer learning for NLP
- Multilingual NLP strategies
- Online serving latency optimization

**Sample Questions:**
- "Design a customer support chatbot. What ML components would you need?"
- "How would you handle multiple languages in an intent detection model?"
- "Explain the tradeoffs between multi-class classification and Q&A-based intent detection"

---

#### 6. Voice Support with ML (2025)
**URL:** https://medium.com/airbnb-engineering/listening-learning-and-helping-at-scale-how-machine-learning-transforms-airbnbs-voice-support-b71f912d4760
**Key Concepts:**
- Domain-adapted ASR (reduced WER from 33% to 10%)
- Contact reason taxonomy and detection
- Help article retrieval with embeddings + LLM re-ranking
- Paraphrasing model for user understanding

**Interview Topics:**
- Speech recognition for domain-specific applications
- Two-stage retrieval (embedding + re-ranking)
- End-to-end voice assistant design

---

### 📊 Causal Inference & Experimentation

#### 7. Artificial Counterfactual Estimation (ACE) (2022)
**URL:** https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
**Key Concepts:**
- ML-based causal inference when A/B tests aren't possible
- Counterfactual prediction using ML models
- Bias correction through A/A tests
- Empirical confidence intervals
- Comparison with propensity score matching, synthetic control

**Interview Topics:**
- Causal inference methods
- When A/B testing isn't possible
- Bias in ML predictions and how to correct it

**Sample Questions:**
- "How would you measure impact when you can't run an A/B test?"
- "Explain the concept of counterfactual estimation"

---

### ⚙️ Feature Engineering & Infrastructure

#### 8. Chronon: Feature Engineering Framework (2023)
**URL:** https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04
**Key Concepts:**
- Declarative feature definition
- Training-serving consistency
- Temporal vs Snapshot accuracy
- Event sources, Entity sources, Cumulative Event sources
- Online (real-time) vs Offline (batch) computation
- Windowed and bucketed aggregations

**Interview Topics:**
- Feature store design
- Training-serving skew prevention
- Real-time vs batch feature computation

**Sample Questions:**
- "How would you design a feature store that guarantees consistency between training and serving?"
- "Explain the challenges of real-time feature computation"

---

### 🔗 Graph ML

#### 9. Graph Machine Learning at Airbnb (2022)
**URL:** https://medium.com/airbnb-engineering/graph-machine-learning-at-airbnb-f868d65f36ee
**Key Concepts:**
- Graph Convolutional Networks (GCN)
- SIGN architecture (Scalable Inception Graph Neural Networks)
- SGC (Simplified GCN) - no trainable weights in GCN layers
- Offline batch computation for graph embeddings
- Trust & safety applications

**Interview Topics:**
- Graph neural network architectures
- When to use graph features
- Scalable graph ML serving

**Sample Questions:**
- "How would you incorporate graph information into an ML model?"
- "Explain the tradeoffs between real-time and batch graph embeddings"

---

### 🖼️ Computer Vision

#### 10. Amenity Detection (2019)
**URL:** https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
**Key Concepts:**
- Object detection for amenity identification
- Image classification for room types
- Multi-label classification

**Interview Topics:**
- Object detection vs image classification
- Multi-label classification approaches

---

### 🏷️ Classification & Categories

#### 11. Building Airbnb Categories with ML and Human-in-the-Loop (2022)
**URL:** https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
**Key Concepts:**
- Rule-based candidate generation (weighted sum of indicators)
- Human-in-the-loop labeling workflow
- Candidate expansion via embedding similarity
- Binary classification for category assignment
- Quality estimation and photo selection models
- Setting thresholds using PR curves (90% precision target)

**Interview Topics:**
- Human-in-the-loop ML systems
- Active learning strategies
- Combining rules and ML

**Sample Questions:**
- "How would you bootstrap an ML model when you have no labeled data?"
- "Explain how to use precision-recall curves to set classification thresholds"

---

### 💰 Pricing & Demand

#### 12. Learning Market Dynamics for Optimal Pricing (2018)
**URL:** https://medium.com/airbnb-engineering/learning-market-dynamics-for-optimal-pricing-97cffbcc53e3
**Key Concepts:**
- Lead time distribution modeling
- Combining ML with structural modeling (Gamma distribution)
- Demand aggregations via listing embeddings and clustering
- Forecasting distribution parameters instead of direct prediction

**Interview Topics:**
- Pricing optimization
- Combining ML with statistical models
- Time series forecasting

**Sample Questions:**
- "How would you model demand for a dynamic pricing system?"
- "When would you combine structural models with ML?"

---

## Cross-Cutting Themes

### Training Data Construction
- User journey/session-based training data (EBR, Embeddings)
- Contrastive learning with positive/negative pairs
- Historical log reconstruction for personalization features
- Human-in-the-loop labeling for categories

### Model Architecture Patterns
- Two-tower networks (retrieval)
- GBDT for tabular ranking
- Transformers for NLP (RoBERTa, XLM-RoBERTa)
- GCN/SIGN for graph features

### Serving Patterns
- Offline embedding computation for listings
- Online query tower computation
- ANN indexes (IVF preferred over HNSW for updateable inventory)
- GPU serving for transformer latency

### Evaluation
- Offline: AUC, NDCG, Precision-Recall curves
- Online: A/B testing, booking metrics
- Causal: ACE when A/B not possible

---

## Interview Preparation Priority

1. **Embedding-Based Retrieval** - Core to search, likely asked
2. **Listing Embeddings** - Foundational, frequently referenced
3. **Search Ranking Evolution** - Shows system design thinking
4. **Feature Engineering (Chronon)** - Training-serving skew is critical
5. **Task-Oriented Conversational AI** - If NLP-focused role
6. **Graph ML** - Emerging area, shows breadth
7. **ACE** - For data science roles
