# ML Interview Tutor 🎓🤖

An interactive AI tutor powered by Claude Code that helps you prepare for machine learning interviews using the Socratic method.

## Features

- **Socratic Method**: Assesses what you know before teaching
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

3. Start learning! Try:
   - "Quiz me on gradient descent"
   - "Explain the bias-variance tradeoff"
   - "Help me understand attention mechanisms"
   - "Practice a ML system design question"

## Repository Structure

```
ml-interview-tutor/
├── CLAUDE.md                    # Instructions for Claude Code
├── README.md                    # This file
├── progress/
│   └── ml-study-tracker.md      # Your progress tracker (single source of truth)
└── sessions/
    ├── SESSION-TEMPLATE.md      # Template for session notes
    └── YYYY-MM-DD/
        └── session-notes.md     # Daily session details
```

## ML Interview Topics Covered

### By Priority (based on interview frequency)

| Domain | Weight | Key Topics |
|--------|--------|------------|
| Deep Learning | 25% | Transformers, attention, CNNs, RNNs, backprop |
| ML Fundamentals | 20% | Gradient descent, bias-variance, regularization |
| ML System Design | 18% | Pipelines, serving, A/B testing, monitoring |
| Classical ML | 15% | Trees, SVM, clustering, logistic regression |
| NLP | 12% | Embeddings, BERT, language models |
| Practical ML | 10% | Debugging, imbalanced data, tuning |

### Detailed Topics

**Fundamentals**: Linear algebra, probability, optimization, loss functions, evaluation metrics

**Classical ML**: Linear/logistic regression, decision trees, random forests, XGBoost, SVM, clustering, PCA

**Deep Learning**: Neural networks, backpropagation, CNNs, RNNs/LSTMs, Transformers, attention mechanisms, normalization

**NLP**: Word embeddings, sequence models, BERT, GPT, fine-tuning

**System Design**: Feature engineering, training pipelines, model serving, A/B testing, monitoring, data drift

**Practical**: Debugging models, handling imbalanced data, hyperparameter tuning, interpretability

## How It Works

1. **Assess**: Asks what you already know about the topic
2. **Explain**: Provides a concise, interview-focused explanation (~200 words)
3. **Verify**: Asks follow-up questions like an interviewer would
4. **Adapt**: Goes deeper or reviews basics based on your answers
5. **Track**: Updates your progress tracker after each session

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
