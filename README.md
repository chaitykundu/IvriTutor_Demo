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

pip install langchain-core langchain-google-genai google-generativeai
pip install google-generativeai cohere pinecone

#📘 RAG Math Tutor (IvriTutor Demo)

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist students with 8th-grade math exercises.
It uses a knowledge base of parsed exercises, Pinecone for vector search, and Google Gemini (Qwen) for natural language understanding.

##Features

Interactive Math Tutor Chatbot with FSM-based flow.

Supports retrieval from JSON exercise bank.

Provides hints, solutions, and answer evaluation.

Uses Pinecone for efficient vector search.

Powered by Google Gemini/Qwen for natural language understanding.

##📋 Prerequisites

Python: Version 3.8 or higher

API Keys:

Pinecone → Get API Key

Google Gemini (Qwen) → Google AI Studio
 or Google Cloud Console

(Optional) Virtual Environment for dependency isolation





