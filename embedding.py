import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re
import unicodedata
from typing import List, Dict
import logging

# ---------- CONFIG ----------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
EMBEDDINGS_OUTPUT_FILE = Path("parsed_outputs/all_parsed_embedding.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text while preserving mathematical notation and structure."""
    try:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        # Preserve Hebrew, English, numbers, and mathematical symbols
        text = re.sub(r'[^\w\s\u0590-\u05FFa-zA-Z0-9$=\-+*^(){}\[\]|<>.,:;]', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def create_pedagogical_chunks(ex: Dict) -> List[Dict]:
    """Create chunks that preserve pedagogical flow and context."""
    chunks = []
    exercise_id = ex.get("canonical_exercise_id", "unknown")
    
    try:
        # Create a comprehensive exercise chunk (main chunk for retrieval)
        grade = ex.get('grade', '')
        lesson = ex.get('lesson_number', '')
        topic = ex.get('topic', '')
        ex_type = ex.get('exercise_type', '')
        ex_number = ex.get('exercise_number', '')
        
        # Main comprehensive chunk - contains the entire exercise context
        questions = ex.get("text", {}).get("question", [])
        solutions = ex.get("text", {}).get("solution", [])
        hints = ex.get("text", {}).get("hint", [])
        full_solutions = ex.get("text", {}).get("full_solution", [])
        
        # Create a complete exercise representation
        main_content = []
        main_content.append(f"שכבה {grade}, שיעור {lesson} - {topic}")
        main_content.append(f"סוג תרגיל: {ex_type}, מספר תרגיל: {ex_number}")
        
        if questions:
            main_content.append("שאלה:")
            main_content.extend(questions)
        
        if full_solutions:
            main_content.append("פתרון מלא:")
            main_content.extend(full_solutions)
        elif solutions:
            main_content.append("פתרון:")
            main_content.extend(solutions)
            
        if hints:
            main_content.append("רמז:")
            main_content.extend(hints)
        
        # Add mathematical expressions for better semantic understanding
        math_expressions = ex.get("math", [])
        if math_expressions:
            main_content.append("ביטויים מתמטיים:")
            main_content.extend(math_expressions)
        
        main_text = "\n".join(main_content)
        
        if main_text.strip():
            chunks.append({
                "exercise_id": exercise_id,
                "chunk_type": "complete_exercise",
                "text": clean_text(main_text),
                "content_type": "pedagogical_unit",
                "retrieval_priority": 1.0  # High priority for main chunks
            })
        
        # Create specific pedagogical chunks for targeted retrieval
        
        # Question chunks (for when student asks about similar questions)
        for i, q in enumerate(questions):
            if q.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "question_only",
                    "text": clean_text(f"שאלה בנושא {topic}: {q}"),
                    "content_type": "question",
                    "retrieval_priority": 0.8
                })
        
        # Hint chunks (for providing guidance)
        for i, hint in enumerate(hints):
            if hint.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "hint_only", 
                    "text": clean_text(f"רמז לתרגיל בנושא {topic}: {hint}"),
                    "content_type": "hint",
                    "retrieval_priority": 0.7
                })
        
        # Solution chunks (for showing solution approaches)
        solution_texts = full_solutions if full_solutions else solutions
        for i, sol in enumerate(solution_texts):
            if sol.strip():
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "solution_only",
                    "text": clean_text(f"פתרון לתרגיל בנושא {topic}: {sol}"),
                    "content_type": "solution",
                    "retrieval_priority": 0.9
                })
        
        # Mathematical expression chunks (for formula similarity)
        for i, math_expr in enumerate(math_expressions):
            if math_expr.strip() and len(math_expr.strip()) > 5:  # Avoid very short expressions
                chunks.append({
                    "exercise_id": exercise_id,
                    "chunk_type": "math_expression",
                    "text": clean_text(f"ביטוי מתמטי בנושא {topic}: {math_expr}"),
                    "content_type": "mathematical_expression",
                    "retrieval_priority": 0.6
                })
                
    except Exception as e:
        logger.error(f"Error processing exercise {exercise_id}: {str(e)}")
    
    return chunks

def main():
    """Main function to process exercises and generate embeddings."""
    try:
        logger.info(f"Loading parsed data from {PARSED_INPUT_FILE}")
        with open(PARSED_INPUT_FILE, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

        all_chunks = []
        for ex in parsed_data:
            chunks = create_pedagogical_chunks(ex)
            all_chunks.extend(chunks)

        # Filter out empty texts and sort by priority
        valid_chunks = [chunk for chunk in all_chunks if chunk["text"].strip()]
        valid_chunks.sort(key=lambda x: x.get("retrieval_priority", 0), reverse=True)
        
        texts = [chunk["text"] for chunk in valid_chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()

        for chunk, emb in zip(valid_chunks, embeddings):
            chunk["embedding"] = emb

        logger.info(f"Saving embeddings to {EMBEDDINGS_OUTPUT_FILE}")
        EMBEDDINGS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully saved {len(valid_chunks)} chunk embeddings")
        
        # Log chunk distribution
        chunk_types = {}
        for chunk in valid_chunks:
            chunk_type = chunk.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        logger.info(f"Chunk distribution: {chunk_types}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()