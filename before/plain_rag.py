"""
BEFORE: plain RAG, prompt-engineering baseline.

Retrieve the top match, hand it to the LLM, answer -- regardless of how
relevant the retrieved match actually is. No confidence check, no
escalation path, no learning from what happens next. If nothing relevant
exists in the knowledge base, the AI answers anyway, using whatever the
closest (possibly irrelevant) match happens to be.

Usage:
    python before/plain_rag.py "Can I change my shipping address after ordering?"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_store import seed_if_empty, query
from llm_client import call_llm

PROMPT_TEMPLATE = """You are a customer support agent. Answer the customer's
question using the reference information below.

Reference information:
{reference}

Customer question: {question}

Answer:"""


def answer(question: str):
    seed_if_empty()
    matches = query(question, n_results=1)
    reference = matches[0]["answer"] if matches else "(no relevant information found)"

    prompt = PROMPT_TEMPLATE.format(reference=reference, question=question)
    response = call_llm(prompt)

    print(f"[before] Retrieved (confidence not checked): {reference[:80]}...")
    print(f"[before] Answered directly, no escalation possible.")
    print(f"[before] Response: {response}")
    return response


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python plain_rag.py "<question>"')
        sys.exit(1)
    answer(sys.argv[1])
