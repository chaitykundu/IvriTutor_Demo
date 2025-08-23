
import json
from pathlib import Path
import cohere
import re
import unicodedata
from typing import List, Dict
import logging

# ---------- CONFIG ----------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
EMBEDDINGS_OUTPUT_FILE = Path("parsed_outputs/all_embeddings.json")
COHERE_API_KEY = "OwF2XbWlNZQAWUPRX7v0Ak5QaRIbbyS6zRWz95PT"  # Replace with your Cohere API key
MODEL_NAME = "embed-multilingual-v3.0"
BATCH_SIZE = 96  # Cohere supports up to 96 texts per call

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text for better embedding quality, handling Hebrew and English characters."""
    try:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        # Preserve Hebrew and English characters
        text = re.sub(r'[^\w\s\u0590-\u05FFa-zA-Z]', '', text)
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def chunk_exercise(ex: Dict) -> List[Dict]:
    """Divide a parsed exercise into semantic chunks."""
    chunks = []
    exercise_id = ex.get("canonical_exercise_id", "unknown")
    try:
        title_text = f"Grade {ex.get('grade', '')}, Lesson {ex.get('lesson_number', '')} - {ex.get('topic', '')}"
        if title_text.strip():
            chunks.append({
                "exercise_id": exercise_id,
                "chunk_type": "title",
                "text": clean_text(title_text)
            })
        for q in ex.get("text", {}).get("question", []):
            if q.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "question",
                    "text": clean_text(q)
                })
        for sol in ex.get("text", {}).get("solution", []):
            if sol.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "solution",
                    "text": clean_text(sol)
                })
        for hint in ex.get("text", {}).get("hint", []):
            if hint.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "hint",
                    "text": clean_text(hint)
                })
        for fs in ex.get("text", {}).get("full_solution", []):
            if fs.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "full_solution",
                    "text": clean_text(fs)
                })
    except Exception as e:
        logger.error(f"Error processing exercise {exercise_id}: {str(e)}")
    return chunks

def main():
    """Main function to process exercises and generate embeddings with Cohere."""
    try:
        logger.info(f"Loading parsed data from {PARSED_INPUT_FILE}")
        with open(PARSED_INPUT_FILE, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        logger.info(f"Initializing Cohere client with model: {MODEL_NAME}")
        co = cohere.Client(COHERE_API_KEY)

        all_chunks = []
        for ex in parsed_data:
            chunks = chunk_exercise(ex)
            all_chunks.extend(chunks)

        texts = [chunk["text"] for chunk in all_chunks if chunk["text"]]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            try:
                response = co.embed(
                    texts=batch_texts,
                    model=MODEL_NAME,
                    input_type="search_document"  # Suitable for storing embeddings
                )
                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//BATCH_SIZE + 1}: {str(e)}")

        for chunk, emb in zip([c for c in all_chunks if c["text"]], embeddings):
            chunk["embedding"] = emb  # Cohere returns lists, no need for .tolist()

        logger.info(f"Saving embeddings to {EMBEDDINGS_OUTPUT_FILE}")
        EMBEDDINGS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully saved {len(all_chunks)} chunk embeddings")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()