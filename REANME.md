# Care2Heal — An Agentic RAG Assistant for Elderly Medication Safety and Care

Care2Heal is an agentic Retrieval-Augmented Generation (RAG) system designed to help
elderly patients and their caregivers manage medications safely — with minimal typing,
multilingual voice support, and built-in emergency detection.

## Problem

Elderly patients often struggle with: understanding jargon-heavy discharge summaries,
tracking multiple medications from different doctors, and catching dangerous drug
interactions before they become harmful. Most existing health apps assume a
tech-comfortable user — Care2Heal is designed for someone who has never used an app before.

## Key Features

- **Photo-to-checklist**: Upload a photo of a prescription or discharge summary — OCR +
  LLM extraction turns it into a structured, plain-language daily medicine checklist
- **Agentic RAG with self-grading**: The agent checks whether its own retrieved context
  is actually relevant before answering, rather than blindly generating a response
- **Drug interaction checking**: Cross-references a patient's full medication list
  against known interactions
- **Emergency detection**: A two-layer safety check (keyword + LLM) flags potential
  emergencies and bypasses the normal Q&A flow entirely
- **Multilingual voice support**: Speak a question in any language (via Whisper) and
  receive spoken responses in your chosen language (Tamil, Hindi, English)
- **QR-based reorder flow**: Scan a code to reorder a medicine, with a tap-to-confirm
  step — never a fully silent automatic purchase

## Architecture

[Insert your architecture diagram screenshot here]

The system is an agent graph (built with LangGraph) that routes each request through:
safety check → (emergency / smalltalk / retrieve) → relevance grading → grounded
answer generation or honest refusal.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Agent framework | LangGraph | Explicit graph structure for branching agent logic |
| Vector DB | ChromaDB | Local, free, no account needed |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, strong semantic search |
| LLM | Groq (openai/gpt-oss-120b) | Free tier, fast inference, strong reasoning |
| OCR | Tesseract | Free, open-source, reliable on printed text |
| Speech-to-text | Whisper | Multilingual out of the box, runs locally |
| Text-to-speech | gTTS | Simple, supports Indian languages |
| Frontend | Streamlit | Fast to build a real multi-page interactive app |

## Known Limitations

- OCR accuracy drops significantly on cursive/stylized fonts — a well-documented
  limitation of Tesseract, partially mitigated by LLM-based typo correction downstream
- Currently seeded with 11 common medications; designed to scale to a full drug
  database (e.g. OpenFDA) without architecture changes
- Reorder flow is simulated (no real payment/pharmacy API integration) for this
  portfolio version — designed with a human-confirmation step to match how a real
  integration should behave safely

## Setup

1. Clone the repo: `git clone https://github.com/yourusername/care2heal-ai.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Install Tesseract OCR separately ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))
5. Add your free Groq API key to a `.env` file: `GROQ_API_KEY=your_key_here`
6. Build the knowledge base: `python data/build_kb.py`
7. Run the app: `streamlit run ui/app.py`

## Demo

## Demo

[Watch the demo video](demo.mp4)