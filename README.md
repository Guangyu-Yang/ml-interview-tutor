# ML Interview Tutor 🎓🤖

An interactive AI tutor powered by Claude Code that helps you prepare for machine learning interviews using the Socratic method.

## Features

- **Structured Quiz Mode**: 3-step workflow (category → depth → topic) for focused practice
- **Article/Paper Quiz Mode**: Bring any article, paper, or tech blog — the tutor extracts key concepts and creates a Socratic quiz from it
- **Socratic Method**: Assesses what you know before teaching
- **Flexible Depth Levels**: Choose high-level concepts, mathematical derivations, or implementation details
- **8 Topic Categories**: From ML fundamentals to LLMs to system design
- **Concise Explanations**: ~200 word explanations that respect your time
- **Active Learning**: Verifies understanding with follow-up questions
- **Adaptive Teaching**: Adjusts difficulty based on your responses
- **Progress Tracking**: Tracks topics mastered, knowledge gaps, and interview readiness
- **Session History**: Maintains detailed notes from each study session

## Getting Started

### Prerequisites

- [Claude Code](https://claude.ai/code) or Claude CLI
- Basic terminal/command line knowledge

### Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Guangyu-Yang/ml-interview-tutor.git
   cd ml-interview-tutor
   ```

2. Open with Claude Code:
   ```bash
   claude
   ```

3. Start learning! Three ways to practice:

   **Option A: Structured Quiz Mode** (Recommended)
   - Type: `"quiz me"` or `"I want to practice"`
   - Choose a category (1-8): ML Fundamentals, Deep Learning, LLMs, System Design, etc.
   - Choose depth level (1-3): High-level, Mathematical, or Implementation
   - Get a tailored question and start learning!

   **Option B: Article/Paper Quiz Mode**
   - Share a link: `"Create a quiz from this article: [URL]"`
   - Or paste content directly: `"Here's an article, quiz me on it: [pasted text]"`
   - Supports any ML-related web content: articles, papers, blog posts, documentation
   - The tutor extracts key concepts, maps them to study domains, and walks through them Socratically

   **Option C: Direct Questions**
   - "Explain the bias-variance tradeoff"
   - "Help me understand attention mechanisms"
   - "Derive the gradient for logistic regression"
   - "Quiz me on AUC-ROC"

## Repository Structure

```
ml-interview-tutor/
├── CLAUDE.md                    # Instructions for Claude Code
├── README.md                    # This file
├── LICENSE                      # MIT license
├── agents/                      # Specialized tutor behaviors
├── hooks/                       # Claude Code hooks
│   └── session-start-suggester.sh
├── progress/
│   └── ml-study-tracker.md      # Your progress tracker (single source of truth)
├── references/                  # Reference materials
└── sessions/
    ├── SESSION-TEMPLATE.md      # Template for session notes
    └── YYYY-MM-DD/
        └── session-notes.md     # Daily session details
```

## ML Interview Topics Covered

### By Priority (based on interview frequency)

| Domain | Weight | Key Topics |
|--------|--------|------------|
| Deep Learning & RL | 25% | Transformers, attention, CNNs, RNNs, policy gradients, RLHF |
| ML Fundamentals | 20% | Gradient descent, bias-variance, regularization |
| ML System Design | 18% | Pipelines, serving, A/B testing, monitoring |
| Classical ML | 15% | Trees, SVM, clustering, logistic regression |
| NLP & Multi-Modal | 12% | Embeddings, BERT, ViT, CLIP, multi-modal LLMs, diffusion |
| Practical ML | 10% | Debugging, imbalanced data, tuning |

### Detailed Topics

**Fundamentals**: Linear algebra, probability, optimization, loss functions, evaluation metrics

**Classical ML**: Linear/logistic regression, decision trees, random forests, XGBoost, SVM, clustering, PCA

**Deep Learning & RL**: Neural networks, backpropagation, CNNs, RNNs/LSTMs, Transformers, attention mechanisms, normalization, RL fundamentals, policy gradients, RLHF/DPO

**NLP & Multi-Modal**: Word embeddings, sequence models, BERT, GPT, fine-tuning, ViT, CLIP, multi-modal LLMs, diffusion models

**System Design**: Feature engineering, training pipelines, model serving, A/B testing, monitoring, data drift

**Practical**: Debugging models, handling imbalanced data, hyperparameter tuning, interpretability

## How It Works

### Structured Quiz Mode (3-Step Workflow)

1. **Choose Category** (8 options):
   - ML Fundamentals, Classical ML, Deep Learning, Transformers & LLMs, NLP, ML System Design, ML Infrastructure, Practical ML

2. **Choose Depth** (3 levels):
   - High-Level Understanding (concepts, intuition, trade-offs)
   - Mathematical Deep Dive (derivations, proofs, complexity)
   - Implementation Details (algorithms, code, architecture)

3. **Practice Topic**: Get a tailored question based on your selections

### Article/Paper Quiz Mode

Bring any external resource and turn it into a study session:

1. **Share a link or paste content** from an article, paper, or tech blog
2. **Tutor analyzes** the material and identifies key technical concepts
3. **Maps to study domains** and generates a focused quiz topic
4. **Walks through Socratically** — doesn't just re-state the article, but guides you to deeply internalize the concepts through questioning

Works with any ML-related web content. If a URL is blocked, just paste the content directly.

### Direct Question Mode

Ask any ML question directly, and the tutor will:

1. **Assess**: Ask what you already know about the topic
2. **Explain**: Provide concise, interview-focused explanation (~200 words)
3. **Verify**: Ask follow-up questions like an interviewer would
4. **Adapt**: Go deeper or review basics based on your answers
5. **Track**: Update your progress tracker after each session

## Progress Tracking

Your progress is tracked in two places:

1. **`/progress/ml-study-tracker.md`** - Overall progress, topics mastered, knowledge gaps, interview readiness
2. **`/sessions/YYYY-MM-DD/session-notes.md`** - Detailed notes from each study session

## Interview Prep Tips

- Practice explaining concepts out loud (interviewers assess communication)
- Focus on intuition first, math second
- Always discuss trade-offs (shows senior thinking)
- Know your projects deeply - expect follow-up questions
- Ask clarifying questions - it's a green flag

## Teaching Philosophy

This tutor follows the Socratic method:
- Questions before answers
- Build on what you already know
- Guide discovery, don't lecture
- Verify understanding before moving on
- Adapt to your level

## License

MIT
