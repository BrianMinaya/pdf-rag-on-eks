#!/usr/bin/env bash
# =============================================================================
# Test RAG Pipeline via CLI
#
# Sends test questions to the Chat API and displays results.
# Assumes kubectl port-forward is running on port 8000.
#
# Usage:
#   # First, in another terminal:
#   kubectl port-forward svc/chat-api 8000:8000 -n pdf-rag-chatbot
#
#   # Then run this script:
#   ./scripts/test-rag.sh
# =============================================================================
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

echo "=== PDF RAG Chatbot -- CLI Test ==="
echo "API URL: ${API_URL}"
echo ""

# --- Health check ---
echo ">>> Health check..."
HEALTH=$(curl -s "${API_URL}/health")
echo "Response: ${HEALTH}"
echo ""

# --- Test question 1 ---
echo ">>> Question 1: What is the main topic of this book?"
RESPONSE=$(curl -s -X POST "${API_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this book?"}')

echo "Answer: $(echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}")"
echo ""

# --- Test question 2 ---
echo ">>> Question 2: Summarize the key concepts discussed."
RESPONSE=$(curl -s -X POST "${API_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you summarize the key concepts discussed in this document?"}')

echo "Answer: $(echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}")"
echo ""

# --- Test question 3 (with history for context) ---
echo ">>> Question 3: Follow-up question (testing conversation history)..."
RESPONSE=$(curl -s -X POST "${API_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you elaborate on the first point?",
    "history": [
      {"role": "user", "content": "What are the main topics?"},
      {"role": "assistant", "content": "The main topics discussed include several key areas."}
    ]
  }')

echo "Answer: $(echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}")"
echo ""

echo "=== Tests complete ==="
