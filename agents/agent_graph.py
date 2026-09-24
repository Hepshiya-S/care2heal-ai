import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agents.rag_pipeline import retrieve_context, llm

load_dotenv()

# --- State: the "memory" that flows through every node in the graph ---
# Every node reads from this and writes back to it. Think of it as a
# shared clipboard passed from station to station.
class AgentState(TypedDict):
    question: str
    documents: List[str]
    metadatas: List[dict]
    relevant: bool
    answer: str


# --- Node 1: Retrieve ---
def retrieve_node(state: AgentState) -> AgentState:
    documents, metadatas = retrieve_context(state["question"])
    state["documents"] = documents
    state["metadatas"] = metadatas
    return state


# --- Node 2: Grade relevance (the "self-check" step) ---
def grade_node(state: AgentState) -> AgentState:
    context_block = "\n\n".join(state["documents"])

    # We ask the LLM a YES/NO question about its OWN retrieval —
    # this is the agent inspecting its own work before trusting it.
    grading_prompt = f"""You are checking whether the CONTEXT below actually contains
information that directly answers the QUESTION. Reply with only one word: "yes" or "no".

Context:
{context_block}

Question: {state['question']}
"""
    response = llm.invoke(grading_prompt)
    verdict = response.content.strip().lower()
    state["relevant"] = verdict.startswith("yes")
    return state


# --- Node 3a: Generate a grounded answer (only reached if relevant) ---
def generate_node(state: AgentState) -> AgentState:
    context_block = "\n\n".join(
        f"[Source: {meta['name']}]\n{doc}"
        for doc, meta in zip(state["documents"], state["metadatas"])
    )
    prompt = f"""You are a careful medical information assistant for elderly patients and their caregivers.

Use ONLY the context below to answer. Do NOT use outside knowledge.

Context:
{context_block}

Question: {state['question']}

Answer in simple, plain language. Mention which source(s) your answer is based on."""
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# --- Node 3b: Honest refusal (only reached if NOT relevant) ---
def insufficient_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "I don't have information on this in my knowledge base. "
        "Please consult a doctor or pharmacist."
    )
    return state


# --- The routing function: this is the actual "decision" ---
# LangGraph calls this after grade_node to decide which path to take next.
def route_after_grading(state: AgentState) -> str:
    return "generate" if state["relevant"] else "insufficient"


# --- Build the graph: nodes + edges, exactly like our flowchart ---
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("generate", generate_node)
graph.add_node("insufficient", insufficient_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")

# This is the branch — "grade" doesn't go to one fixed next step,
# it calls route_after_grading() and goes wherever that returns.
graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "generate": "generate",
        "insufficient": "insufficient"
    }
)

graph.add_edge("generate", END)
graph.add_edge("insufficient", END)

# Compile turns our node/edge definitions into a runnable agent
app = graph.compile()


if __name__ == "__main__":
    print("Care2Heal Agentic RAG (Day 4) — type a question (or 'quit')\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "quit":
            break
        result = app.invoke({
            "question": user_question,
            "documents": [],
            "metadatas": [],
            "relevant": False,
            "answer": ""
        })
        print(f"\nCare2Heal: {result['answer']}\n")