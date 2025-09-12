import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from dialogue import DialogueFSM
import json

# Load environment variables
load_dotenv(dotenv_path=Path(".env"))

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
INDEX_NAME = "mathtutor-e5-large"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# -----------------------------
# Helper Functions
# -----------------------------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not PARSED_INPUT_FILE.exists():
        logger.error("❌ Missing JSON file.")
        return

    try:
        exercises = load_json(PARSED_INPUT_FILE)
        pinecone_index = get_pinecone_index()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"❌ Error loading data or connecting to Pinecone: {e}")
        return

    # Initialize GenAI Chat Client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in .env")
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Initialize DialogueFSM
    fsm = DialogueFSM(exercises, pinecone_index, llm)
    initial_response = fsm.transition("")
    print(f"A_GUY: {initial_response}")

    while True:
        try:
            fsm.inactivity_timer.mark_typing()
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            fsm.inactivity_timer.stop()
            break
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            fsm.inactivity_timer.stop()
            break

        response = fsm.transition(user_input)
        print(f"A_GUY: {response}")

if __name__ == "__main__":
    main()