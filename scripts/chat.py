#!/usr/bin/env python3
"""
Interactive CLI Chat Client for the PDF RAG Chatbot.

This script provides a terminal-based interface for chatting with the RAG
pipeline. It connects to the Chat API (FastAPI) and sends questions via
POST /chat, displaying the LLM's answers and source citations with
formatted ANSI colors.

HOW IT WORKS:
1. On startup, the script checks the API's /health endpoint to verify
   the server is reachable.
2. It enters an interactive input loop where you type questions.
3. Each question is sent to POST /chat along with the conversation history
   (so the LLM understands follow-up questions).
4. The response is displayed with the answer in bold and sources in dim text.
5. The conversation history grows with each exchange, enabling multi-turn
   conversations just like a real chat.

CONVERSATION HISTORY:
The client manages history locally in a Python list. Each exchange adds two
entries: {"role": "user", "content": "..."} and {"role": "assistant",
"content": "..."}. This mirrors the OpenAI chat message format that the
Chat API expects. The full history is sent with every request so the LLM
has context for follow-up questions.

Usage:
    # First, port-forward the Chat API:
    kubectl port-forward svc/chat-api 8000:8000 -n pdf-rag-chatbot

    # Then run this script:
    python scripts/chat.py

    # Or override the API URL:
    API_URL=http://my-server:9000 python scripts/chat.py

Special commands:
    quit / exit  -- End the session
    clear        -- Reset conversation history (start fresh)
"""

import os
import sys

import httpx

# ---------------------------------------------------------------------------
# ANSI color codes for terminal formatting
# ---------------------------------------------------------------------------
# These escape codes tell the terminal to change text styling. They work on
# macOS Terminal, iTerm2, and most Linux terminals. On Windows, they work in
# Windows Terminal and PowerShell 7+.
#
# The format is: \033[<code>m  where <code> controls the style.
# \033[0m resets all formatting back to normal.
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def health_check(client: httpx.Client) -> bool:
    """Verify the Chat API is reachable before entering the chat loop."""
    try:
        resp = client.get(f"{API_URL}/health")
        resp.raise_for_status()
        return True
    except httpx.HTTPError as e:
        print(f"{RED}Health check failed: {e}{RESET}")
        return False


def display_sources(sources: list[dict]) -> None:
    """Print source citations with page numbers and similarity scores."""
    if not sources:
        return
    print(f"\n{DIM}--- Sources ---{RESET}")
    for i, src in enumerate(sources, 1):
        score = src.get("score", 0)
        page = src.get("page_number", "?")
        filename = src.get("source", "unknown")
        # Truncate long source text to keep the terminal readable.
        text = src.get("text", "")
        preview = text[:120].replace("\n", " ") + ("..." if len(text) > 120 else "")
        print(f"{DIM}  [{i}] p.{page} ({score:.2%} match) {filename}{RESET}")
        print(f"{DIM}      {preview}{RESET}")


def main() -> None:
    # Use a persistent HTTP client so TCP connections are reused across
    # requests (connection pooling). The 120s timeout matches the Chat API's
    # own timeout for vLLM calls -- LLM generation can be slow.
    client = httpx.Client(timeout=120.0)

    print(f"{BOLD}{CYAN}=== PDF RAG Chatbot -- CLI ==={RESET}")
    print(f"API: {API_URL}")
    print()

    # --- Health check: make sure the server is up ---
    print("Checking API health...", end=" ")
    if not health_check(client):
        print(f"\n{RED}Could not reach the Chat API at {API_URL}{RESET}")
        print("Make sure the server is running and port-forwarded:")
        print("  kubectl port-forward svc/chat-api 8000:8000 -n pdf-rag-chatbot")
        sys.exit(1)
    print(f"{GREEN}OK{RESET}")
    print()
    print(f"Type your questions below. Commands: {BOLD}clear{RESET} (reset history), {BOLD}quit{RESET} (exit)")
    print()

    # Conversation history in OpenAI chat message format. This is sent with
    # every request so the LLM can understand follow-up questions.
    history: list[dict] = []

    while True:
        try:
            question = input(f"{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C -- exit gracefully without a traceback.
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        # Skip empty input.
        if not question:
            continue

        # Handle special commands.
        if question.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye!{RESET}")
            break
        if question.lower() == "clear":
            history.clear()
            print(f"{YELLOW}Conversation history cleared.{RESET}\n")
            continue

        # --- Send the question to the Chat API ---
        try:
            resp = client.post(
                f"{API_URL}/chat",
                json={
                    "question": question,
                    "history": history if history else None,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            print(f"{RED}Error: {e}{RESET}\n")
            continue

        # --- Display the answer ---
        answer = data.get("answer", "(no answer)")
        print(f"\n{BOLD}{CYAN}Bot:{RESET} {answer}")

        # --- Display sources ---
        display_sources(data.get("sources", []))

        # --- Update conversation history ---
        # Add both the user's question and the assistant's answer so the LLM
        # has full context for follow-up questions.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        print()


if __name__ == "__main__":
    main()
