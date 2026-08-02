import sys
import os
import json
import hashlib
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END

from knowledge_store import seed_if_empty, query, add_learned_entry
from llm_client import call_llm
from state import SupportState

HUMAN_ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "..", "human_sim", "human_answers.json")
NEEDS_REVIEW_PATH = os.path.join(os.path.dirname(__file__), "..", "needs_review.json")

CONFIDENCE_THRESHOLD = 0.55

RESPONSE_PROMPT_TEMPLATE = """You are a customer support agent. Answer the
customer's question using the reference information below. Be concise.

Reference information:
{reference}

Customer question: {question}

Answer:"""

REFRAME_PROMPT_TEMPLATE = """You are validating a human support agent's answer
before it becomes a permanent, reusable knowledge base entry that will
answer future customers automatically, without a human involved.

Customer question: {question}
Human's raw answer: {raw_answer}

Rewrite this into a clear, standalone, general answer suitable for reuse.
If the human's answer is too vague, incomplete, specific to this one
customer's situation (an order number, a name, a one-off exception), or
otherwise not something that should be reused as a general answer,
respond with EXACTLY:
VAGUE: <one sentence reason>

Otherwise respond with EXACTLY:
GOOD: <the reframed, reusable answer>
"""


def receive_query(state: SupportState) -> SupportState:
    seed_if_empty()
    state["confidence_threshold"] = CONFIDENCE_THRESHOLD
    print(f"[after] Received query: '{state['question']}'")
    return state


def retrieve(state: SupportState) -> SupportState:
    matches = query(state["question"], n_results=1)
    if matches:
        state["retrieved_answer"] = matches[0]["answer"]
        state["retrieved_question"] = matches[0]["question"]
        state["confidence"] = matches[0]["confidence"]
    else:
        state["retrieved_answer"] = ""
        state["retrieved_question"] = ""
        state["confidence"] = 0.0
    print(f"[after] Retrieved closest match (confidence={state['confidence']:.2f}): "
          f"'{state['retrieved_question']}'")
    return state


def confidence_gate(state: SupportState) -> str:
    """Routing-only. Verification point 1: is the retrieved match good
    enough to answer from directly, or does this need a human?"""
    if state["confidence"] >= state["confidence_threshold"]:
        return "respond_directly"
    return "escalate_to_human"


def respond_directly(state: SupportState) -> SupportState:
    prompt = RESPONSE_PROMPT_TEMPLATE.format(
        reference=state["retrieved_answer"],
        question=state["question"],
    )
    state["final_answer"] = call_llm(prompt)
    state["route"] = "direct"
    state["human_answered"] = False
    print(f"[after] Confidence above threshold, answered directly.")
    return state


def escalate_to_human(state: SupportState) -> SupportState:
    """Simulated human-in-the-loop. Looks up a canned answer from
    human_sim/human_answers.json . Falls back to a real
    input() prompt if no canned answer exists, so you can actually play
    the human role live."""
    state["route"] = "escalated"
    print(f"[after] Confidence below threshold ({state['confidence']:.2f} < "
          f"{state['confidence_threshold']}). Escalating to a human.")

    canned_answers = {}
    if os.path.exists(HUMAN_ANSWERS_PATH):
        with open(HUMAN_ANSWERS_PATH) as f:
            canned_answers = json.load(f)

    if state["question"] in canned_answers:
        raw_answer = canned_answers[state["question"]]
        print(f"[after] (using canned human answer for reproducibility)")
    else:
        raw_answer = input(f"[HUMAN NEEDED] Customer asked: '{state['question']}'\n"
                            f"Your answer: ")

    state["raw_human_answer"] = raw_answer
    state["human_answered"] = True
    return state


def reframe_and_validate(state: SupportState) -> SupportState:
    """Verification point 2. A separate pass reviews the human's raw
    answer before it's allowed anywhere near permanent memory. This is
    the maker/checker split applied to a human's output instead of an
    LLM's: the human isn't writing a reusable FAQ entry, they're
    answering one customer, so something else has to decide if what they
    said generalizes."""
    prompt = REFRAME_PROMPT_TEMPLATE.format(
        question=state["question"],
        raw_answer=state["raw_human_answer"],
    )
    print(f"[after] [validator] Reviewing human answer for reuse...")
    verdict = call_llm(prompt).strip()

    if verdict.upper().startswith("GOOD"):
        state["answer_quality"] = "good"
        state["reframed_answer"] = verdict.split(":", 1)[1].strip()
        state["quality_reason"] = ""
        state["final_answer"] = state["reframed_answer"]
        print(f"[after] [validator] GOOD -- reframed for reuse.")
    else:
        state["answer_quality"] = "vague"
        reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
        state["quality_reason"] = reason
        state["reframed_answer"] = ""
        state["final_answer"] = state["raw_human_answer"]
        print(f"[after] [validator] VAGUE -- {reason}")

    return state


def quality_gate(state: SupportState) -> str:
    """Routing-only. Good answers get embedded into permanent memory.
    Vague ones get logged for a human to look at later."""
    if state["answer_quality"] == "good":
        return "embed_and_store"
    return "log_for_review"


def embed_and_store(state: SupportState) -> SupportState:
    entry_id = "learned-" + hashlib.sha256(state["question"].encode()).hexdigest()[:12]
    add_learned_entry(
        question=state["question"],
        answer=state["reframed_answer"],
        entry_id=entry_id,
    )
    return state


def log_for_review(state: SupportState) -> SupportState:
    """Vague or overly specific human answers don't get embedded. They go
    into a separate file for a person to periodically clean up, decide to
    generalize, or discard -- kept out of the vector DB so one-off or
    unclear answers don't get served to future customers as if they were
    settled FAQ entries."""
    entry = {
        "question": state["question"],
        "raw_human_answer": state["raw_human_answer"],
        "reason": state["quality_reason"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    existing = []
    if os.path.exists(NEEDS_REVIEW_PATH):
        with open(NEEDS_REVIEW_PATH) as f:
            existing = json.load(f)
    existing.append(entry)
    with open(NEEDS_REVIEW_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[after] Logged to needs_review.json instead of the knowledge base.")
    return state


def respond_with_human_answer(state: SupportState) -> SupportState:
    print(f"[after] Response (from human): {state['final_answer']}")
    return state


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("receive_query", receive_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond_directly", respond_directly)
    graph.add_node("escalate_to_human", escalate_to_human)
    graph.add_node("reframe_and_validate", reframe_and_validate)
    graph.add_node("embed_and_store", embed_and_store)
    graph.add_node("log_for_review", log_for_review)
    graph.add_node("respond_with_human_answer", respond_with_human_answer)

    graph.set_entry_point("receive_query")
    graph.add_edge("receive_query", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        confidence_gate,
        {"respond_directly": "respond_directly", "escalate_to_human": "escalate_to_human"},
    )
    graph.add_edge("respond_directly", END)
    graph.add_edge("escalate_to_human", "reframe_and_validate")
    graph.add_conditional_edges(
        "reframe_and_validate",
        quality_gate,
        {"embed_and_store": "embed_and_store", "log_for_review": "log_for_review"},
    )
    graph.add_edge("embed_and_store", "respond_with_human_answer")
    graph.add_edge("log_for_review", "respond_with_human_answer")
    graph.add_edge("respond_with_human_answer", END)

    return graph.compile()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python support_loop.py "<question>"')
        sys.exit(1)

    app = build_graph()
    initial_state: SupportState = {
        "question": sys.argv[1],
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
    final_state = app.invoke(initial_state)

    print(json.dumps(
        {
            "question": final_state["question"],
            "route": final_state["route"],
            "confidence": round(final_state["confidence"], 2),
            "answer_quality": final_state["answer_quality"],
            "final_answer": final_state["final_answer"],
        },
        indent=2,
    ))
