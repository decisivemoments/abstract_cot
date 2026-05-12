from __future__ import annotations


def render_direct_answer_prompt(question: str) -> str:
    return (
        "Solve the following problem and give the final answer only.\n\n"
        f"{question.strip()}\n"
    )


def render_natural_cot_prompt(question: str) -> str:
    return (
        "Solve the following problem step by step, then give the final answer clearly.\n\n"
        f"{question.strip()}\n"
    )


def render_abstract_answer_prompt(question: str, abstract_trace: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Abstract reasoning:\n{abstract_trace.strip()}\n\n"
        "Answer:\n"
    )

