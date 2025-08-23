import os
from pinecone import Pinecone
import cohere

INDEX_NAME = "math-exercises-multilingual"
MODEL_NAME = "embed-multilingual-v3.0"

def main():
    # Initialize Pinecone client using os.getenv
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    pc = Pinecone(api_key=pinecone_api_key)

    # Initialize Cohere client using os.getenv
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set")
    co = cohere.Client(api_key=cohere_api_key)

    index = pc.Index(INDEX_NAME)

    query_text = "נקודות על מערכת צירים עם משולש"  # Hebrew query: "Points on a coordinate system with a triangle"
    # Encode the query text using Cohere's embed method with input_type
    vec = co.embed(texts=[query_text], model=MODEL_NAME, input_type="search_query").embeddings[0]

    # Optional metadata filter: only grade ז and has SVG
    flt = {
        "grade": {"$eq": "ז"},
        "svg_exists": {"$eq": True}
    }

    # Query the Pinecone index
    res = index.query(vector=vec, top_k=5, include_metadata=True, filter=flt)
    for m in res["matches"]:
        print(round(m["score"], 4), m["id"], m["metadata"].get("chunk_type"), m["metadata"].get("topic"))

if __name__ == "__main__":
    main()