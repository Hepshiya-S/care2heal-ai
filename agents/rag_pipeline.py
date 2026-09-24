import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load the GROQ_API_KEY from .env into the environment
load_dotenv()

# --- Load our two "memory" systems ---
# 1. The embedding model: turns text into numbers so we can search by meaning
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. ChromaDB: our vector database from Day 2
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")

# --- Set up the LLM ---
# temperature=0.2 keeps answers factual and consistent rather than creative —
# for medical information, we want the LLM to be conservative, not imaginative.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


def retrieve_context(query: str, n_results: int = 3):
    """
    Takes a user question, finds the most relevant chunks from ChromaDB.
    Returns both the text (for the prompt) and metadata (for citations).
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return documents, metadatas


def build_prompt(query: str, documents: list, metadatas: list) -> str:
    """
    Combines retrieved chunks into a grounded prompt.
    This is the exact mechanism that prevents hallucination.
    """
    context_block = "\n\n".join(
        f"[Source: {meta['name']}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )

    prompt = f"""You are a careful medical information assistant for elderly patients and their caregivers.

Use ONLY the context below to answer the question. Do NOT use any outside knowledge,
even general medical knowledge you may know. If the context does not contain enough
information to answer the question, respond with EXACTLY this and nothing else:
"I don't have information on this in my knowledge base. Please consult a doctor or pharmacist."

Context:
{context_block}

Question: {query}

Answer in simple, plain language a non-medical person can understand. Mention which
source(s) your answer is based on."""

    return prompt


def ask(query: str) -> str:
    """
    The full RAG pipeline: retrieve -> build prompt -> ask LLM -> return answer.
    """
    documents, metadatas = retrieve_context(query)
    prompt = build_prompt(query, documents, metadatas)
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Quick manual test loop
    print("Care2Heal RAG test — type a question (or 'quit' to exit)\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "quit":
            break
        answer = ask(user_question)
        print(f"\nCare2Heal: {answer}\n")