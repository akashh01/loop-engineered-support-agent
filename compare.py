"""
Runs the same previously-unknown question through the loop twice:
  1. First time: not in the knowledge base, gets escalated to a human,
     the human's answer is validated/reframed, then embedded.
  2. Second time: same question, now answered directly by the AI, no
     escalation, because of what the loop learned the first time.

Usage:
    python compare.py "Can I change my shipping address after ordering?"
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from after.support_loop import build_graph, CONFIDENCE_THRESHOLD


def run_query(app, question):
    state = {
        "question": question,
        "retrieved_answer": "",
        "retrieved_question": "",
        "confidence": 0.0,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "route": "direct",
        "raw_human_answer": "",
        "reframed_answer": "",
        "answer_quality": "",
        "quality_reason": "",
        "final_answer": "",
        "human_answered": False,
    }
    return app.invoke(state)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python compare.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]
    app = build_graph()

    print("=== FIRST PASS (before this knowledge exists) ===")
    first = run_query(app, question)
    print(json.dumps({"route": first["route"], "confidence": round(first["confidence"], 2)}, indent=2))

    print("\n=== SECOND PASS (same question, after learning) ===")
    second = run_query(app, question)
    print(json.dumps({"route": second["route"], "confidence": round(second["confidence"], 2)}, indent=2))

    print("\n=== SUMMARY ===")
    print(f"First pass route:  {first['route']} (confidence {first['confidence']:.2f})")
    print(f"Second pass route: {second['route']} (confidence {second['confidence']:.2f})")
    if first["route"] == "escalated" and second["route"] == "direct":
        print("Confirmed: the loop learned from the escalation and answered directly next time.")
