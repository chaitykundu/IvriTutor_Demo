# 📘 RAG Math Tutor (IvriTutor Demo)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** designed to assist students with **8th-grade math exercises**.  
It uses a knowledge base of parsed exercises, **Pinecone** for vector search, and **Google Gemini (Qwen)** for natural language understanding.  

## 🚀 Features
- Interactive **Math Tutor Chatbot** with FSM-based flow  
- Supports **retrieval from JSON exercise bank**  
- Provides **hints, solutions, and answer evaluation**  
- Uses **Pinecone** for efficient vector search  
- Powered by **Google Gemini/Qwen** for natural language understanding  

## 📋 Prerequisites
- **Python**: Version 3.8 or higher  
- **API Keys**:  
  - Pinecone → [Get API Key](https://www.pinecone.io/)  
  - Google Gemini (Qwen) → [Google AI Studio](https://makersuite.google.com/) or [Google Cloud Console](https://console.cloud.google.com/)  
- (Optional) **Virtual Environment** for dependency isolation  

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/IvriTutor_Demo.git
cd IvriTutor_Demo

2. Create a Virtual Environment (Recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

## 4. Configure Environment Variables

Create a .env file in the project root:

PINECONE_API_KEY=your_actual_pinecone_api_key_here
GEMINI_API_KEY=your_actual_gemini_api_key_here

📚 Preparing the Knowledge Base

Place parsed exercise JSON files in:

IvriTutor_Demo/parsed_outputs/


Generate embeddings (first run or after data changes):

python IvriTutor_Demo/parsed_outputs/embedding.py


Index embeddings into Pinecone:

python IvriTutor_Demo/parsed_outputs/index_embedding.py


⚠️ Make sure the index name in index_embedding.py matches the one used in chatbot.py (default: mathtutor-e5-large).

💬 Running the Chatbot

Make sure your virtual environment is active, then run:

## python chatbot.py

Interaction Flow:

The chatbot greets you

You select grade and topic

It presents math exercises

You can:

Provide an answer → chatbot evaluates

Ask for a hint

Request the solution







