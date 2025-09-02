
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Generator
from pinecone import Pinecone, ServerlessSpec
import logging
import numpy as np
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()  # Add this line

# ---------- CONFIG ----------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
EMB_INPUT_FILE = Path("parsed_outputs/all_parsed_embedding.json")
INDEX_NAME = "mathtutor-e5-large" # Changed index name to reflect new model
EMBED_DIM = 1024  # multilingual-e5-large has 1024 dimensions
CLOUD = "aws"
REGION = "us-east-1"  # Must match Pinecone serverless region
BATCH_SIZE = 100  # Pinecone recommends batches of 100-1000 for upserts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- HELPERS ----------
def load_json(p: Path) -> List[Dict[str, Any]]:
    """Load JSON file with error handling."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {p}: {str(e)}")
        raise FileNotFoundError(f"Missing or invalid {p}")

def build_meta_lookup(parsed_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map exercise_id → metadata fields with enhanced information."""
    try:
        lut = {}
        for ex in parsed_records:
            ex_id = ex["canonical_exercise_id"]
            lut[ex_id] = {
                "grade": ex.get("grade", "unknown"),
                "topic": ex.get("topic", "unknown"),
                "lesson_number": ex.get("lesson_number", "unknown"),
                "exercise_type": ex.get("exercise_type", "unknown"),
                "exercise_number": ex.get("exercise_number", "unknown"),
                "difficulty": ex.get("exercise_type", "unknown"),  # Map exercise_type to difficulty
                "svg_exists": bool(ex.get("svg", [])),
                "has_hints": bool(ex.get("text", {}).get("hint", [])),
                "mathematical_concept": ex.get("topic", "unknown")
            }
        logger.info(f"Built metadata lookup for {len(lut)} exercises")
        return lut
    except Exception as e:
        logger.error(f"Error building metadata lookup: {str(e)}")
        raise

def validate_embedding(embedding: List[float], dimension: int = EMBED_DIM) -> bool:
    """Validate that the embedding is a list of correct length and contains valid floats."""
    try:
        if not isinstance(embedding, list) or len(embedding) != dimension:
            logger.warning(f"Invalid embedding dimension: expected {dimension}, got {len(embedding) if isinstance(embedding, list) else 'not a list'}")
            return False
        if not all(isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x) for x in embedding):
            logger.warning("Embedding contains invalid values (NaN or Inf)")
            return False
        return True
    except Exception as e:
        logger.warning(f"Error validating embedding: {str(e)}")
        return False

def ensure_index(pc: Pinecone, index_name: str, dim: int):
    """Create or recreate index with correct dimension."""
    try:
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        if index_name in existing_indexes:
            # Check existing index dimension
            index_desc = pc.describe_index(index_name)
            if index_desc["dimension"] != dim:
                logger.warning(f"Index {index_name} has dimension {index_desc['dimension']}, expected {dim}. Deleting and recreating.")
                pc.delete_index(index_name)
                pc.create_index(
                    name=index_name,
                    dimension=dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=CLOUD, region=REGION)
                )
                logger.info(f"Recreated Pinecone index: {index_name} with dimension {dim}")
            else:
                logger.info(f"Index {index_name} already exists with correct dimension {dim}")
        else:
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION)
            )
            logger.info(f"Created Pinecone index: {index_name}")
        # Wait until index is ready
        pc.describe_index(index_name)
    except Exception as e:
        logger.error(f"Error managing Pinecone index: {str(e)}")
        raise

def batch(iterable: List, n: int = BATCH_SIZE) -> Generator[List, Any, Any]:   
    """Yield batches of size n."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

# ---------- MAIN ----------
def main():
    """Index embeddings in Pinecone with metadata."""
    try:
        # Load data
        parsed_records = load_json(PARSED_INPUT_FILE)
        all_chunks = load_json(EMB_INPUT_FILE)
        meta_lut = build_meta_lookup(parsed_records)

        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set PINECONE_API_KEY environment variable.")
        pc = Pinecone(api_key=api_key)
        ensure_index(pc, INDEX_NAME, EMBED_DIM)
        index = pc.Index(INDEX_NAME)

        # Build vectors with metadata
        vectors = []
        skipped_chunks = 0
        for i, ch in enumerate(all_chunks):
            ex_id = ch["exercise_id"]
            emb = ch.get("embedding", [])
            if not validate_embedding(emb):
                logger.warning(f"Skipping chunk {ex_id}_{ch.get('chunk_type', 'unknown')} due to invalid or missing embedding")
                skipped_chunks += 1
                continue
            md = meta_lut.get(ex_id, {})

            # Generate a safe ASCII ID
            hash_prefix = hashlib.md5(ex_id.encode("utf-8")).hexdigest()[:12]
            vec_id = f"{hash_prefix}::{i:06d}"

            # Enhanced metadata with pedagogical information
            metadata = {
                "exercise_id": ex_id,
                "chunk_type": ch.get("chunk_type", "unknown"),
                "content_type": ch.get("content_type", "unknown"),
                "grade": md.get("grade", "unknown"),
                "topic": md.get("topic", "unknown"),
                "lesson_number": md.get("lesson_number", "unknown"),
                "exercise_type": md.get("exercise_type", "unknown"),
                "exercise_number": md.get("exercise_number", "unknown"),
                "difficulty": md.get("difficulty", "unknown"),
                "mathematical_concept": md.get("mathematical_concept", "unknown"),
                "svg_exists": md.get("svg_exists", False),
                "has_hints": md.get("has_hints", False),
                "retrieval_priority": ch.get("retrieval_priority", 0.5),
                "text": ch.get("text", "")[:2000]  # Increased truncation limit for better context
            }

            vectors.append({
                "id": vec_id,
                "values": emb,
                "metadata": metadata
            })

        # Upsert in batches
        total = 0
        logger.info(f"Indexing {len(vectors)} chunks in Pinecone (skipped {skipped_chunks} invalid chunks)...")
        for chunk in batch(vectors):
            try:
                index.upsert(vectors=chunk)
                total += len(chunk)
                logger.info(f"Upserted {total}/{len(vectors)} vectors")
            except Exception as e:
                logger.error(f"Error upserting batch {total//BATCH_SIZE + 1}: {str(e)}")
                continue

        # Verify index stats
        stats = index.describe_index_stats()
        logger.info(f"Pinecone index stats: {stats}")
        if skipped_chunks > 0:
            logger.warning(f"Skipped {skipped_chunks} chunks. Regenerate embeddings with a valid API key.")
        logger.info(f"✅ Finished upserting {total} vectors into '{INDEX_NAME}'.")

        # Log indexing summary
        chunk_type_counts = {}
        content_type_counts = {}
        for vector in vectors:
            chunk_type = vector["metadata"].get("chunk_type", "unknown")
            content_type = vector["metadata"].get("content_type", "unknown")
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        logger.info(f"Chunk type distribution: {chunk_type_counts}")
        logger.info(f"Content type distribution: {content_type_counts}")

    except Exception as e:
        logger.error(f"Error in indexing process: {str(e)}")
        raise

if __name__ == "__main__":
    main()