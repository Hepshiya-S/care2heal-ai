import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load our curated medical data
with open("data/drug_data.json", "r") as f:
    drugs = json.load(f)

# Load the embedding model (downloads once, then cached locally)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a persistent ChromaDB client — "persistent" means data
# is saved to disk in data/chroma_db/, so it survives app restarts.
# Without this, you'd re-build the whole database every time you run the app.
client = chromadb.PersistentClient(path="data/chroma_db")

# Create (or get) a collection — think of this like a table in a database
collection = client.get_or_create_collection(name="medical_knowledge")

# Turn each drug entry into a searchable text chunk.
# We combine all fields into one descriptive paragraph per drug —
# this way, a semantic search for "swelling in feet" can match
# Amlodipine even though the user didn't say "Amlodipine" or "side effects."
documents = []
metadatas = []
ids = []

for drug in drugs:
    text_chunk = (
        f"Drug name: {drug['name']}. Category: {drug['category']}. "
        f"Dosage: {drug['dosage']}. Side effects: {drug['side_effects']}. "
        f"Interacts with: {', '.join(drug['interactions'])}. "
        f"Explanation: {drug['plain_explanation']}"
    )
    documents.append(text_chunk)
    metadatas.append({
        "name": drug["name"],
        "source": drug["source"]
    })
    ids.append(drug["id"])

# Generate embeddings for all chunks at once (more efficient than one-by-one)
print(f"Embedding {len(documents)} drug entries...")
embeddings = model.encode(documents).tolist()

# Store everything in ChromaDB: the text, its embedding, metadata, and ID
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"Knowledge base built successfully with {len(documents)} entries.")