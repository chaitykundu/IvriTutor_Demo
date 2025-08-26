##How It Works (Pipeline)

Ingest & Normalize → JSON → structured format.
Embed & Index → semantic chunks stored in vector DB.
Dialogue FSM → orchestrates user flow.
Retrieve → get top-k similar exercises by metadata.
Generate Variant → LLM creates solvable Hebrew/English exercise.

##Tech Stack

Vector DB: Pinecone
Embeddings: Cohere embedding
LLM: Gemini, Cohere, or Hugging Face models
Math Verification: SymPy (CAS checks)
Geometry Validation: Custom logic for SVG parameters


