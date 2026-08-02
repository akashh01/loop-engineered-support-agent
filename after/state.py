from typing import TypedDict, Literal


class SupportState(TypedDict):
    question: str
    retrieved_answer: str
    retrieved_question: str
    confidence: float
    confidence_threshold: float
    route: Literal["direct", "escalated"]
    raw_human_answer: str
    reframed_answer: str
    answer_quality: str  # "good" | "vague" | ""
    quality_reason: str
    final_answer: str
    human_answered: bool
