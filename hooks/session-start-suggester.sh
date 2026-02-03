#!/bin/bash
# Session Start Hook: Proactive Study Suggestions
# This hook runs when Claude Code starts and provides context about what to study next

PROGRESS_FILE="/Users/guangyuyang/projects/ml-interview-tutor/progress/ml-study-tracker.md"

# Check if progress file exists
if [[ ! -f "$PROGRESS_FILE" ]]; then
    echo "Welcome to ML Interview Tutor! No progress tracked yet."
    echo "Would you like to: 1) Start a quiz, 2) Ask a question, or 3) See the study plan?"
    exit 0
fi

# Read progress file
PROGRESS=$(cat "$PROGRESS_FILE")

# Extract key information
READINESS=$(echo "$PROGRESS" | grep "Overall Interview Readiness" | grep -oE '[0-9]+%' | head -1)
LAST_UPDATED=$(echo "$PROGRESS" | grep "Last Updated" | head -1 | sed 's/.*: //')

# Extract topic counts per domain
DL_PROGRESS=$(echo "$PROGRESS" | grep "C. Deep Learning" | grep -oE '[0-9]+ \| [0-9]+' | head -1)
FUND_PROGRESS=$(echo "$PROGRESS" | grep "A. ML Fundamentals" | grep -oE '[0-9]+ \| [0-9]+' | head -1)
SYSDES_PROGRESS=$(echo "$PROGRESS" | grep "E. ML System Design" | grep -oE '[0-9]+ \| [0-9]+' | head -1)

# Extract upcoming topics (first 3 unchecked items)
UPCOMING=$(echo "$PROGRESS" | grep -A10 "### Upcoming Topics" | grep "^\- \[ \]" | head -3 | sed 's/- \[ \] /  - /')

# Extract knowledge gaps
GAPS=$(echo "$PROGRESS" | grep -A5 "### .*Priority" | grep "^-" | grep -v "None" | head -2 | sed 's/^/  /')

# Build the suggestion output
echo "<session-start-context>"
echo "ML Interview Tutor - Session Start"
echo ""
echo "Progress Summary:"
echo "  - Interview Readiness: $READINESS"
echo "  - Last Session: $LAST_UPDATED"
echo "  - Deep Learning: $DL_PROGRESS topics"
echo "  - ML Fundamentals: $FUND_PROGRESS topics"
echo "  - System Design: $SYSDES_PROGRESS topics"
echo ""

if [[ -n "$GAPS" && ! "$GAPS" =~ "None" ]]; then
    echo "Knowledge Gaps to Address:"
    echo "$GAPS"
    echo ""
fi

if [[ -n "$UPCOMING" ]]; then
    echo "Recommended Next Topics:"
    echo "$UPCOMING"
    echo ""
fi

echo "Suggested Actions:"
echo "  1. Continue study plan (next topic)"
echo "  2. Quiz me (structured practice)"
echo "  3. Review a mastered topic"
echo "  4. Ask a specific question"
echo "  5. Full system design practice"
echo ""
echo "What would you like to do today?"
echo "</session-start-context>"

exit 0
