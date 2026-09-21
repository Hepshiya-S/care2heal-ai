import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="medical_knowledge")

# Simulate a real user query — notice we're NOT using the drug's exact name,
# to prove semantic search is working, not just keyword matching
query = "my feet are swelling, what medicine could cause that?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("Query:", query)
print("\nTop matches:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n- Source: {meta['name']} ({meta['source']})")
    print(f"  {doc}")