import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agents.rag_pipeline import retrieve_context, llm
from modules.safety_checker import check_emergency

load_dotenv()

GREETING_KEYWORDS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "thank you", "thanks", "bye", "goodbye", "ok", "okay"
]


def is_smalltalk(text: str) -> bool:
    """Quick keyword check for greetings/casual chat — avoids wasting an LLM
    call on something this simple, and keeps 'hello' from being treated as
    a failed medical lookup."""
    cleaned = text.lower().strip().strip("!.,?")
    return cleaned in GREETING_KEYWORDS or any(
        cleaned.startswith(kw + " ") for kw in GREETING_KEYWORDS
    )


class AgentState(TypedDict):
    question: str
    documents: List[str]
    metadatas: List[dict]
    relevant: bool
    is_emergency: bool
    is_smalltalk: bool
    answer: str


def safety_check_node(state: AgentState) -> AgentState:
    state["is_smalltalk"] = is_smalltalk(state["question"])
    # No need to run emergency detection on obvious small talk
    state["is_emergency"] = False if state["is_smalltalk"] else check_emergency(state["question"])
    return state


def smalltalk_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "Hello! I'm Care2Heal, here to help with your medicines and health questions. "
        "You can ask me things like \"What does Metformin do?\" or \"What should I be careful of with my medicines?\""
    )
    return state


def emergency_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "⚠ This sounds like it may be a medical emergency. "
        "Please call your local emergency number immediately or contact a caregiver right away. "
        "This app cannot provide emergency medical care."
    )
    return state


def retrieve_node(state: AgentState) -> AgentState:
    documents, metadatas = retrieve_context(state["question"])
    state["documents"] = documents
    state["metadatas"] = metadatas
    return state


def grade_node(state: AgentState) -> AgentState:
    context_block = "\n\n".join(state["documents"])
    grading_prompt = f"""You are checking whether the CONTEXT below contains information
that is USEFUL for answering the QUESTION, even if it doesn't answer it completely or
mentions a related medicine rather than an exact match. Reply with only one word: "yes" or "no".

Context:
{context_block}

Question: {state['question']}
"""
    response = llm.invoke(grading_prompt)
    verdict = response.content.strip().lower()
    state["relevant"] = verdict.startswith("yes")
    return state


def generate_node(state: AgentState) -> AgentState:
    context_block = "\n\n".join(
        f"[Source: {meta['name']}]\n{doc}"
        for doc, meta in zip(state["documents"], state["metadatas"])
    )
    prompt = f"""You are a careful medical information assistant for elderly patients and their caregivers.

Use ONLY the context below to answer. Do NOT use outside knowledge, even general medical
knowledge you may know. If the context does not contain enough information, respond with
EXACTLY: "I don't have information on this in my knowledge base. Please consult a doctor or pharmacist."

Context:
{context_block}

Question: {state['question']}

Answer in simple, plain language. Mention which source(s) your answer is based on."""
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


def insufficient_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "I don't have information on this in my knowledge base. "
        "Please consult a doctor or pharmacist."
    )
    return state


def route_after_safety(state: AgentState) -> str:
    if state["is_emergency"]:
        return "emergency"
    if state["is_smalltalk"]:
        return "smalltalk"
    return "retrieve"


def route_after_grading(state: AgentState) -> str:
    return "generate" if state["relevant"] else "insufficient"


graph = StateGraph(AgentState)

graph.add_node("safety_check", safety_check_node)
graph.add_node("emergency", emergency_node)
graph.add_node("smalltalk", smalltalk_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("generate", generate_node)
graph.add_node("insufficient", insufficient_node)

graph.set_entry_point("safety_check")

graph.add_conditional_edges(
    "safety_check",
    route_after_safety,
    {
        "emergency": "emergency",
        "smalltalk": "smalltalk",
        "retrieve": "retrieve"
    }
)

graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "generate": "generate",
        "insufficient": "insufficient"
    }
)

graph.add_edge("emergency", END)
graph.add_edge("smalltalk", END)
graph.add_edge("generate", END)
graph.add_edge("insufficient", END)

app = graph.compile()


if __name__ == "__main__":
    print("Care2Heal Agentic RAG — type a question (or 'quit')\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "quit":
            break
        result = app.invoke({
            "question": user_question,
            "documents": [],
            "metadatas": [],
            "relevant": False,
            "is_emergency": False,
            "is_smalltalk": False,
            "answer": ""
        })
        print(f"\nCare2Heal: {result['answer']}\n")