# chatbot_new.py - 4-State Math Tutoring System (Enhanced with all original features)
# Import necessary components from the LangChain and Python libraries.
import os
import json
import random
import re
import time
import threading
import logging
import uuid
import google.generativeai as genai
from pathlib import Path
from enum import Enum, auto
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(dotenv_path=Path(".env"))

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Pinecone Config
INDEX_NAME = "mathtutor-e5-large"
EMBED_DIM = 1024
TOP_K_RETRIEVAL = 20

# Embedding Model Config
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Enhanced Inactivity Settings
INACTIVITY_TIMEOUT = 60  # Increased from 30 seconds
TYPING_DETECTION_THRESHOLD = 10  # Seconds to wait for complete input

# Progressive Guidance Settings
MIN_ATTEMPTS_BEFORE_HINT = 1
MIN_ATTEMPTS_BEFORE_SOLUTION = 2
MAX_GUIDANCE_LEVELS = 3  # 0=encouragement, 1=guiding_question, 2=hint, 3=solution

# --- 1. SET UP THE LLM AND API KEY ---
# Replace 'your-api-key' with your actual Google API key.
# It's best practice to use environment variables for API keys.
os.environ["GOOGLE_API_KEY"] = "AIzaSyAdKYyRoN1R7G9KERy2HuQZ2Pabs4mSOkY"
from enum import Enum, auto
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(dotenv_path=Path(".env"))

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Pinecone Config
INDEX_NAME = "mathtutor-e5-large"
EMBED_DIM = 1024
TOP_K_RETRIEVAL = 20

# Embedding Model Config
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Enhanced Inactivity Settings
INACTIVITY_TIMEOUT = 60  # Increased from 30 seconds
TYPING_DETECTION_THRESHOLD = 10  # Seconds to wait for complete input

# Progressive Guidance Settings
MIN_ATTEMPTS_BEFORE_HINT = 1
MIN_ATTEMPTS_BEFORE_SOLUTION = 2
MAX_GUIDANCE_LEVELS = 3  # 0=encouragement, 1=guiding_question, 2=hint, 3=solution

# -----------------------------
# GenAI Setup
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# -----------------------------
# GenAI Setup (Enhanced with both simple and advanced configurations)
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Initialize the LLM. We'll use the ChatGoogleGenerativeAI model with 'gemini-2.5-flash'.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Enhanced RAG prompt for context-based learning
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Math AI tutor. 
    
    Language Rules:
    - If the question is in Hebrew → respond in Hebrew
    - If the question is in English → respond in English
    - Always match the user's language preference
    - For Hebrew responses, use Right-to-Left (RTL) formatting for conversational text.
    - Ensure all mathematical expressions and scientific notation remain Left-to-Right (LTR), even within Hebrew sentences.
    
    Teaching Guidelines:
    - Never give direct answers immediately
    - Use a gradual assistance approach: encouragement → guiding questions → hints → solution
    - Ask guiding questions to help students think through problems
    - Build understanding step by step
    - Use the provided context to give accurate information
    - If context lacks crucial information, state what's missing
    - When providing hints, use EXACT TEXT from context when available
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context: {context}\n\nQuestion: {input}")
])
rag_chain = rag_prompt | llm

# Enhanced prompt templates
small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor starting a conversation.
    
    Language Rules:
    - Detect the user's language and respond in the same language
    - If Hebrew is detected, respond in Hebrew
    - If English is detected, respond in English
    - Default to English if language is unclear
    
    Personality:
    - Warm, encouraging, and approachable
    - Enthusiastic about helping with math
    - Keep responses short and conversational (1-2 sentences max)
    - Understand the user's intent even with spelling mistakes or unclear input
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
small_talk_chain = small_talk_prompt | llm

# Personal follow-up prompt (bilingual)
personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are continuing a casual conversation with a student.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - Keep the same language throughout the conversation
    
    Guidelines:
    - Acknowledge their response warmly
    - Keep it brief and natural (1-2 sentences)
    - Show genuine interest in personal topics like work, sports, daily life
    - Gradually transition toward academic readiness
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
personal_followup_chain = personal_followup_prompt | llm

# Academic transition prompt (bilingual)
academic_transition_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are transitioning from personal chat to academic topics.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - For Hebrew: Use proper RTL formatting for general text, keep math expressions LTR
    
    Academic Transition Guidelines:
    - Ask about recent learning or upcoming academic events
    - Examples: "What did you learn recently?", "When is your next exam?", "How's school going?", "What subjects are you studying?"
    - Bridge from personal to academic naturally
    - Keep it friendly but start showing academic interest
    - Keep responses short (1 sentence)
    - Make the transition feel natural
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
academic_transition_chain = academic_transition_prompt | llm

# --- 3. CREATE THE PROMPT TEMPLATE ---
# The prompt is now updated to instruct the AI to be a helpful assistant
# in both English and Hebrew. This tells the LLM to detect the user's
# language and respond accordingly.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Math AI assistant. You can converse in both English and Hebrew. Please respond in the same language as the user's question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# --- 4. CREATE THE CHAIN ---
# A chain links the prompt to the LLM, creating a single, callable unit.
chain = prompt | llm

# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "inactivity_check": "Are you still there? I'm here to help whenever you're ready!",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "irrelevant_msg": "I can only help with math exercises and related questions.",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "guiding_question": "🤔 Let me ask you this: ",
        "encouragement": "You're making progress — give it a try first!",
        "try_again": "Can you try again? Think about your approach.",
    },
    "he": {
        "inactivity_check": "אתה עדיין כאן? אני כאן לעזור בכל עת שתהיה מוכן!",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "irrelevant_msg": "אני יכול לעזור רק עם תרגילי מתמטיקה ושאלות קשורות.",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "guiding_question": "🤔 תן לי לשאול אותך את זה: ",
        "encouragement": "אתה מתקדם - תנסה קודם!",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
    }
}

# -----------------------------
# 4-State Structure
# -----------------------------
class State(Enum):
    STATE_1_OPENING = auto()
    STATE_2_DIAGNOSTIC = auto()
    STATE_3_LEARNING = auto()
    STATE_4_SUMMARY = auto()

# -----------------------------
# Helper Functions
# -----------------------------
def detect_language(text: str) -> str:
    """Detect if text is Hebrew or English."""
    if any('\u0590' <= char <= '\u05FF' for char in text):
        return "he"
    return "en"

def clean_math_text(text: str) -> str:
    """Remove LaTeX-style $ signs and other delimiters from math expressions."""
    if not text:
        return text
    # Remove inline LaTeX ($...$)
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
    # Remove display LaTeX ($$...$$)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    # Remove backslash commands (e.g., \frac, \sqrt) but keep content
    text = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', text)
    # Remove standalone backslashes
    text = re.sub(r'\\([a-zA-Z]+)', r'', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining $ or $$ that might be malformed
    text = text.replace('$', '')
    return text.strip()

def translate_text_to_english(text: str) -> str:
    """Translate text (likely Hebrew) to English using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise translator. Translate the following text to English. If it's already in English, return it as is. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = response.content.strip()
        translated = clean_math_text(translated)
        return translated
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return text

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# Load embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using SentenceTransformer."""
    if embedding_model is None:
        logger.error("Embedding model not loaded.")
        return []
    try:
        return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_chunks(query: str, pc_index: Any, grade: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from Pinecone based on a query."""
    query = clean_math_text(query)
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []

    filter_dict = {}
    if grade:
        filter_dict["grade"] = {"$eq": grade}
    if topic:
        filter_dict["topic"] = {"$eq": topic}

    try:
        response = pc_index.query(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVAL,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return [match.metadata for match in response.matches]
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}", exc_info=True)
        return []

def describe_svg_content(svg_content: str) -> str:
    """Describe SVG content using GenAI."""
    try:
        svg_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code. Provide only a brief description."),
            ("user", "Describe the following SVG content:\n```svg\n{svg_input}\n```"),
        ])
        svg_description_chain = svg_description_prompt | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return clean_math_text(response.content)
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."

def clean_math_text(text: str) -> str:
    """Remove LaTeX-style $ signs and other delimiters from math expressions."""
    if not text:
        return text
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', text)
    text = re.sub(r'\\([a-zA-Z]+)', r'', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('$', '')
    return text.strip()

def translate_text_to_english(text: str) -> str:
    """Translate text to English using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise translator. Translate the following text to English. If it's already in English, return it as is. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = response.content.strip()
        return clean_math_text(translated)
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return text

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# Load embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using SentenceTransformer."""
    if embedding_model is None:
        logger.error("Embedding model not loaded.")
        return []
    try:
        return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_chunks(query: str, pc_index: Any, grade: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from Pinecone based on a query."""
    query = clean_math_text(query)
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []

    filter_dict = {}
    if grade:
        filter_dict["grade"] = {"$eq": grade}
    if topic:
        filter_dict["topic"] = {"$eq": topic}

    try:
        response = pc_index.query(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVAL,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return [match.metadata for match in response.matches]
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}", exc_info=True)
        return []

# -----------------------------
# Enhanced Inactivity Timer with Typing Detection
# -----------------------------
class EnhancedInactivityTimer:
    def __init__(self, callback, timeout=INACTIVITY_TIMEOUT):
        self.callback = callback
        self.timeout = timeout
        self.timer = None
        self.last_activity_time = time.time()
        self.typing_detected = False
        
    def start(self):
        self.stop()
        self.timer = threading.Timer(self.timeout, self._check_inactivity)
        self.timer.daemon = True
        self.timer.start()
        
    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
            
    def reset(self):
        self.last_activity_time = time.time()
        self.typing_detected = False
        self.start()
    
    def mark_typing(self):
        """Mark that user is typing - prevents premature timeout."""
        self.typing_detected = True
        self.last_activity_time = time.time()
    
    def _check_inactivity(self):
        """Check if user is truly inactive."""
        current_time = time.time()
        time_since_activity = current_time - self.last_activity_time
        
        if self.typing_detected and time_since_activity < TYPING_DETECTION_THRESHOLD:
            # User was typing recently, extend timer
            self.start()
        elif time_since_activity >= self.timeout:
            # Truly inactive
            self.callback()
        else:
            # Restart timer for remaining time
            remaining_time = self.timeout - time_since_activity
            self.timer = threading.Timer(remaining_time, self._check_inactivity)
            self.timer.daemon = True
            self.timer.start()

def describe_svg_content(svg_content: str) -> str:
    """Describe SVG content using GenAI."""
    try:
        svg_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant who is professional mathematician. Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code. Provide only a brief description."),
            ("user", "Describe the following SVG content:\n```svg\n{svg_input}\n```"),
        ])
        svg_description_chain = svg_description_prompt | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return clean_math_text(response.content)
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."

# -----------------------------
# Enhanced Attempt Tracking
# -----------------------------
class AttemptTracker:
    def __init__(self):
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
        
    def reset(self):
        """Reset for new question."""
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
    
    def record_attempt(self, is_correct: bool):
        """Record an attempt and return if guidance should be offered."""
        self.total_attempts += 1
        if not is_correct:
            self.incorrect_attempts += 1
        
        return not is_correct and self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT
    
    def can_provide_hint(self) -> bool:
        """Check if hint can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT or 
                self.has_requested_hint)
    
    def can_provide_solution(self) -> bool:
        """Check if solution can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_SOLUTION or
                self.has_requested_solution or
                (self.has_requested_hint and self.incorrect_attempts >= 1))
    
    def should_encourage_more_attempts(self, is_hint_request: bool = False, is_solution_request: bool = False) -> bool:
        """Determine if we should encourage more attempts instead of giving help."""
        if is_solution_request and not self.can_provide_solution():
            return True
        if is_hint_request and not self.can_provide_hint():
            return True
        return False

# -----------------------------
# Main 4-State FSM
# -----------------------------
class MathTutorFSM:
    def __init__(self, exercises_data, pinecone_index):
        self.state = State.STATE_1_OPENING
        self.exercises_data = exercises_data
        self.pinecone_index = pinecone_index
        self.chat_history = []
        self.user_language = "en"
        
        # State tracking
        self.opening_step = 0
        self.diagnostic_answers = {}
        self.exercise_counter = 0  # Track completed exercises (max 2)
        self.lesson_complete = False  # Track if lesson is finished
        self.current_exercise = None
        self.current_question_index = 0
        self.guidance_level = 0
        
        # Exercise state
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 5
        
        # User context from opening
        self.user_hobby = None
        self.user_day_info = None
        
        # SVG handling attributes
        self.current_svg_description = None
        self.current_svg_file_path = None
        self.svg_generated_for_question = False
        
        # Progressive guidance tracking
        self.attempt_tracker = AttemptTracker()
        
        # Inactivity timer
        self.inactivity_timer = EnhancedInactivityTimer(self._handle_inactivity)
        self._start_inactivity_timer()

    def _start_inactivity_timer(self):
        """Start or reset the inactivity timer."""
        self.inactivity_timer.reset()

    def _handle_inactivity(self):
        """Handle inactivity timeout - EXACT specification: 'If no response for 60 seconds → send nudge message'."""
        # EXACT specification requirement: "Hey, are you still there?"
        nudge_message = "Hey, are you still there?"
        self._send_inactivity_message(nudge_message)
    
    def _send_inactivity_message(self, message):
        """Send inactivity message."""
        print(f"\n[INACTIVITY TIMEOUT] A_GUY: {message}")

    def _get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        return text.format(**kwargs) if kwargs else text

    def _get_localized_text(self, key: str, fallback: str = None, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N.get(self.user_language, I18N["en"])
        text = lang_dict.get(key, fallback or I18N["en"].get(key, key))
        return text.format(**kwargs) if kwargs else text

    def _detect_inappropriate_content(self, text: str) -> bool:
        """Detect inappropriate content."""
        blacklist = ["sex", "violence", "drugs", "politics", "religion"]
        return any(word in text.lower() for word in blacklist)

    def _detect_humor(self, text: str) -> bool:
        """Detect humor as per specification: 'haha', 'lol', or laughter emojis 😂"""
        text = text.lower()
        humor_keywords = ['haha', 'lol', 'ahaha', 'hahaha', 'ahahaha']
        laughter_emojis = ['😂', '🤣', '😆', '😄', '😀', '😃', '😁']
        
        # Check for humor keywords
        for keyword in humor_keywords:
            if keyword in text:
                return True
        
        # Check for laughter emojis
        for emoji in text:
            if emoji in laughter_emojis:
                return True
        
        return False

    def _generate_personalized_followup(self, hobby: str) -> str:
        """Generate personalized follow-up question based on hobby."""
        try:
            followup_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly tutor. Generate a personalized follow-up question about the hobby: {hobby}.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Ask something specific about their hobby
                - Show genuine interest
                - Keep it conversational and brief (1 sentence)
                - Examples: "Did you play basketball today?", "How long have you been playing guitar?"
                """),
                ("user", f"The student's hobby is: {hobby}")
            ])
            
            followup_chain = followup_prompt | llm
            response = followup_chain.invoke({})
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating personalized follow-up: {e}")
            return "That's interesting! Tell me more about it."

    def _generate_humorous_reaction(self, context: str) -> str:
        """Generate humorous reaction based on context."""
        try:
            humor_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly tutor making a light, humorous comment.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Make a brief, light-hearted comment
                - Connect to transitioning to math/learning
                - Keep it friendly and encouraging
                - 1 sentence maximum
                - Example: "Great! Let's see if your brain is as fit as your legs."
                """),
                ("user", f"Context: {context}")
            ])
            
            humor_chain = humor_prompt | llm
            response = humor_chain.invoke({})
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating humorous reaction: {e}")
            return "Great! Let's see if your brain is ready for some math."

    def _generate_natural_response_to_how_are_you(self, user_input: str) -> str:
        """Generate natural response to user's 'how are you' answer, then ask about their day."""
        try:
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI tutor responding naturally to the student's answer about how they are.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - First, respond naturally to their answer about how they are
                - If they ask about your day, give a brief, friendly response (e.g., "I'm doing great, thanks for asking!")
                - Then smoothly transition to asking about their day
                - Keep it warm and conversational
                - End with asking: "How was your day? Long day?"
                - Maximum 2-3 sentences total
                
                Example flow:
                User: "Good, and yours?"
                You: "I'm doing great, thanks for asking! How was your day? Long day?"
                """),
                ("user", f"Student's response to 'how are you': {user_input}")
            ])
            
            chain = response_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            if "your" in user_input.lower() or "you" in user_input.lower():
                return "I'm doing great, thanks for asking! How was your day? Long day?"
            else:
                return "That's nice to hear! How was your day? Long day?"
        except Exception as e:
            logger.error(f"Error generating natural response to how are you: {e}")
            # Fallback response
            if "your" in user_input.lower() or "you" in user_input.lower():
                return "I'm doing great, thanks for asking! How was your day? Long day?"
            else:
                return "That's nice to hear! How was your day? Long day?"

    def _generate_natural_response_to_day_info(self, user_input: str) -> str:
        """Generate natural response to user's day info, then ask about hobbies."""
        try:
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI tutor responding naturally to the student's answer about their day.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - First, respond naturally and empathetically to what they shared about their day
                - Show interest and understanding
                - Then smoothly transition to asking about hobbies
                - End with asking: "What hobbies do you have?"
                - Keep it warm and conversational
                - Maximum 2-3 sentences total
                
                Examples:
                If they had a good day: "That sounds wonderful!"
                If they had a tough day: "I understand, some days can be challenging."
                If they had a long day: "Long days can be tiring!"
                """),
                ("user", f"Student's response about their day: {user_input}")
            ])
            
            chain = response_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "That sounds interesting! What hobbies do you have?"
        except Exception as e:
            logger.error(f"Error generating natural response to day info: {e}")
            # Fallback response
            return "That sounds interesting! What hobbies do you have?"

    def _wants_to_chat_more(self, user_input: str) -> bool:
        """Detect if user wants to chat more before proceeding to diagnostic."""
        chat_indicators = [
            "talk more", "chat more", "can we talk", "let's chat", "tell me more",
            "before we start", "wait", "hold on", "can we", "let's", "tell me",
            "share", "discuss", "conversation", "more about", "can we discuss",
            "daily conversations", "about daily", "more discussion", "keep talking",
            "continue chatting", "want to discuss", "let's discuss more", "chat about",
            "can i talk", "talk a bit", "talk about myself", "about myself", "tell you about",
            "share about myself", "get to know", "know each other", "more personal"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in chat_indicators)

    def _handle_chat_request_before_diagnostic(self, user_input: str) -> str:
        """Handle user's request to chat more before diagnostic questions."""
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI tutor who wants to chat a bit more before starting the lesson.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge their desire to chat more positively
                - Be warm and encouraging about getting to know them
                - Gently transition to explaining why you need to ask a few questions
                - Explain that the questions help you understand their needs better
                - End with asking the first diagnostic question: "Do you have a test coming up?"
                - Keep it natural and conversational
                - Maximum 2-3 sentences
                
                Example responses:
                "I'd love to chat more! Let me ask you a few quick questions first so I can help you better. Do you have a test coming up?"
                "Sure, I enjoy getting to know my students! These questions will help me understand what you need. Do you have a test coming up?"
                """),
                ("user", f"Student wants to chat more: {user_input}")
            ])
            
            chain = chat_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "I'd love to chat more! Let me ask you a few quick questions first so I can help you better. Do you have a test coming up?"
        except Exception as e:
            logger.error(f"Error handling chat request: {e}")
            # Fallback response
            return "I'd love to chat more! Let me ask you a few quick questions first so I can help you better. Do you have a test coming up?"

    def _is_general_conversation_request(self, user_input: str) -> bool:
        """Detect general conversation requests that need natural responses."""
        conversation_indicators = [
            "can we talk", "let's talk", "tell me about", "what about you", 
            "how about you", "what do you think", "do you like", "have you",
            "can you tell me", "i want to know", "what's your", "share with me",
            "but can we", "before we", "can we discuss", "let's discuss",
            "daily conversations", "about daily", "more discussion", "keep talking",
            "continue chatting", "want to discuss", "no work just chatting",
            "just chatting", "want to chat", "prefer to talk", "rather chat",
            "can i talk", "talk a bit", "talk about myself", "about myself", "tell you about",
            "share about myself", "get to know", "know each other", "more personal"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in conversation_indicators)

    def _wants_to_skip_to_math(self, user_input: str) -> bool:
        """Detect if user wants to skip small talk and go directly to math."""
        math_indicators = [
            "math", "mathematics", "math part", "math please", "skip to math",
            "let's do math", "start math", "begin math", "math lesson",
            "math problems", "exercises", "questions", "skip", "let's start",
            "get started", "start now", "begin", "lesson", "problems",
            "מתמטיקה", "חשבון", "תרגילים", "שאלות", "בואו נתחיל"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in math_indicators)

    def _handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation requests naturally while steering towards lesson."""
        try:
            conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor who wants to be conversational but also needs to guide the lesson.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Current state: {self.state.name}
                
                Guidelines:
                - Acknowledge their conversational request warmly
                - Give a brief, natural response to their question/comment
                - Gently redirect back to the lesson structure
                - Be encouraging and friendly
                - If they're asking about you, be personable but brief
                - If in diagnostic state, guide towards the diagnostic questions
                - If in learning state, guide back to the math problems
                - Maximum 2-3 sentences
                
                Examples:
                "I appreciate you wanting to chat! I'd love to get to know you better as we work together. Let me ask you a few questions to understand how I can help you best."
                "That's a great question! I enjoy connecting with my students. Now, let's focus on helping you with math - shall we continue with where we left off?"
                """),
                ("user", f"Student's conversational request: {user_input}")
            ])
            
            chain = conversation_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            if self.state == State.STATE_2_DIAGNOSTIC:
                return "I'd love to chat more! Let me ask you a few quick questions first so I can help you better. Do you have a test coming up?"
            elif self.state == State.STATE_3_LEARNING:
                return "I appreciate you wanting to talk! Let's focus on this math problem first, then we can chat more."
            else:
                return "That's interesting! I'd love to get to know you better as we work together."
        except Exception as e:
            logger.error(f"Error handling general conversation: {e}")
            # Fallback response based on current state
            if self.state == State.STATE_2_DIAGNOSTIC:
                return "I'd love to chat more! Let me ask you a few quick questions first so I can help you better. Do you have a test coming up?"
            elif self.state == State.STATE_3_LEARNING:
                return "I appreciate you wanting to talk! Let's focus on this math problem first, then we can chat more."
            else:
                return "That's interesting! I'd love to get to know you better as we work together."

    def _handle_skip_to_math_request(self, user_input: str) -> str:
        """Handle user's request to skip small talk and move to math."""
        try:
            skip_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor. The student wants to skip the small talk and get straight to math.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge their enthusiasm for math positively
                - Be understanding that they want to get started right away
                - Transition smoothly to the diagnostic questions
                - Keep it brief and positive
                - End by asking the first diagnostic question: "Do you have a test coming up?"
                - Maximum 2 sentences
                
                Examples:
                "I love your enthusiasm! Let's dive right into it. Do you have a test coming up?"
                "Perfect, let's get started with math! Do you have a test coming up?"
                """),
                ("user", f"Student wants to skip to math: {user_input}")
            ])
            
            chain = skip_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "I love your enthusiasm! Let's dive right into it. Do you have a test coming up?"
        except Exception as e:
            logger.error(f"Error handling skip to math request: {e}")
            return "I love your enthusiasm! Let's dive right into it. Do you have a test coming up?"

    def _is_session_ending_request(self, user_input: str) -> bool:
        """Detect if user wants to end the session."""
        ending_indicators = [
            "let's end", "end for today", "stop here", "that's enough", "finish now",
            "i'm done", "enough for today", "call it a day", "wrap up", "stop now",
            "no more", "i want to stop", "let's stop", "finish this", "done for today"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in ending_indicators)

    def _handle_session_ending_request(self, user_input: str) -> str:
        """Handle user's request to end the session."""
        try:
            ending_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor. The student wants to end the session.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge their request respectfully
                - Be understanding and supportive
                - Briefly mention what we've accomplished so far
                - Offer to continue another time
                - Be encouraging and positive
                - Maximum 2-3 sentences
                - Transition to summary if appropriate
                
                Example: "I understand you'd like to wrap up! We've made good progress today. Let me give you a quick summary of what we covered."
                """),
                ("user", f"Student wants to end session: {user_input}")
            ])
            
            chain = ending_prompt | llm
            response = chain.invoke({})
            
            # Transition to summary after this response
            self.state = State.STATE_4_SUMMARY
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            self.state = State.STATE_4_SUMMARY
            return "I understand you'd like to wrap up! Let me give you a quick summary of what we covered today."
        except Exception as e:
            logger.error(f"Error handling session ending request: {e}")
            # Fallback response and transition to summary
            self.state = State.STATE_4_SUMMARY
            return "I understand you'd like to wrap up! Let me give you a quick summary of what we covered today."

    def _handle_learning_conversation_request(self, user_input: str) -> str:
        """Handle conversation requests during learning state."""
        try:
            learning_conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor in the middle of a learning session. The student wants to have a conversation.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge their conversational request warmly
                - Show interest but gently redirect to the current math problem
                - Be encouraging about staying focused
                - Mention that we can chat more after solving the current exercise
                - Maximum 2-3 sentences
                - Keep it friendly but focused on learning
                
                Example: "I'd love to hear more about you! Let's finish this math problem first, then we can chat more. Can you take another look at the exercise?"
                """),
                ("user", f"Student's conversation request during learning: {user_input}")
            ])
            
            chain = learning_conversation_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "I'd love to hear more about you! Let's finish this math problem first, then we can chat more. Can you take another look at the exercise?"
        except Exception as e:
            logger.error(f"Error handling learning conversation request: {e}")
            return "I'd love to hear more about you! Let's finish this math problem first, then we can chat more. Can you take another look at the exercise?"

    def _is_unclear_response(self, user_input: str) -> bool:
        """Detect unclear or minimal responses that need clarification."""
        unclear_indicators = ["?", "??", "???", "huh", "unclear", "confused", "i don't understand", "what do you mean"]
        
        user_input = user_input.strip()
        user_lower = user_input.lower()
        
        # Don't treat practice requests as unclear
        if "practice" in user_lower or "more" in user_lower:
            return False
            
        # Don't treat questions about timing as unclear if they contain "now" or "when"
        if "now" in user_lower or "when" in user_lower:
            return False
        
        # Check for very short responses
        if len(user_input) <= 2 and user_input in ["?", "??", "wat", "hm", "um", "eh"]:
            return True
        
        # Check for standalone "what" questions (but not "what about now" type questions)
        if user_lower.strip() == "what" or user_lower.strip() == "what?":
            return True
            
        # Check for unclear indicators (but be more specific about "what")
        return any(indicator in user_lower for indicator in unclear_indicators if indicator != "what")

    def _handle_unclear_response(self, user_input: str) -> str:
        """Handle unclear responses based on current state."""
        if self.state == State.STATE_2_DIAGNOSTIC:
            if len(self.diagnostic_answers) == 0:
                return "I was asking if you have a test coming up. Do you have any math tests or quizzes soon?"
            elif len(self.diagnostic_answers) == 1:
                return "I'd like to know what you covered in your last math class. What topics did your teacher go over recently?"
            elif len(self.diagnostic_answers) == 2:
                return "I'm asking what specific math topic you'd like to work on today. Is there anything particular you're struggling with?"
        
        elif self.state == State.STATE_3_LEARNING:
            return "I can see you might be confused about the math problem. Would you like me to give you a hint, or shall we break it down step by step?"
        
        elif self.state == State.STATE_1_OPENING:
            if self.opening_step == 1:
                return "I was asking how you're doing today. How are you feeling?"
            elif self.opening_step == 2:
                return "I'd like to know about your day. Was it a good day or a challenging one?"
            elif self.opening_step == 3:
                return "I'm curious about your hobbies. What do you like to do in your free time?"
        
        # Default response
        return "I want to make sure I understand you correctly. Could you tell me a bit more about what you mean?"

    def _wants_more_practice(self, user_input: str) -> bool:
        """Detect if user wants more practice."""
        practice_indicators = [
            "practice more", "more practice", "can we practice", "want to practice",
            "let's practice", "continue practicing", "keep practicing", "another problem",
            "more problems", "more exercises", "another exercise", "keep going",
            "continue", "more", "again", "what about now", "now", "right now"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in practice_indicators)

    def _handle_more_practice_request(self, user_input: str) -> str:
        """Handle user's request for more practice."""
        try:
            practice_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor. The student wants to practice more after the lesson.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Be enthusiastic about their desire to practice more
                - Acknowledge that practice is great for learning
                - Explain that this session is complete but they can start a new session
                - Suggest they can restart the conversation for more practice
                - Be positive and encouraging
                - Maximum 2-3 sentences
                
                Example: "I love your enthusiasm for more practice! This session is complete, but you can definitely start a new conversation with me anytime for more math problems. Just say hello again and we'll dive into more practice!"
                """),
                ("user", f"Student wants more practice: {user_input}")
            ])
            
            chain = practice_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "I love your enthusiasm for more practice! This session is complete, but you can start a new conversation with me anytime for more math problems."
        except Exception as e:
            logger.error(f"Error handling more practice request: {e}")
            return "I love your enthusiasm for more practice! This session is complete, but you can start a new conversation with me anytime for more math problems."

    def _is_goodbye(self, user_input: str) -> bool:
        """Detect goodbye messages."""
        goodbye_indicators = ["bye", "goodbye", "see you", "farewell", "thanks", "thank you", "later"]
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in goodbye_indicators)

    def _handle_summary_conversation(self, user_input: str) -> str:
        """Handle general conversation in summary state."""
        try:
            summary_conversation_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly AI math tutor in the summary/closing phase of a lesson. The student wants to have a conversation.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Be warm and conversational since the lesson is ending
                - Answer their question/respond to their comment naturally
                - Keep it brief and positive
                - You can be more relaxed since the formal lesson is complete
                - Maximum 2-3 sentences
                
                Example: "That's a great question! I'm glad you're thinking about that. You've shown such good curiosity today!"
                """),
                ("user", f"Student's conversation in summary state: {user_input}")
            ])
            
            chain = summary_conversation_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "That's a great point! I really enjoyed our session today. You did fantastic work!"
        except Exception as e:
            logger.error(f"Error handling summary conversation: {e}")
            return "That's a great point! I really enjoyed our session today. You did fantastic work!"

    def _pick_exercise_based_on_diagnostic(self) -> bool:
        """Pick an exercise based on diagnostic answers."""
        upcoming_test = self.diagnostic_answers.get('test', '')
        last_class = self.diagnostic_answers.get('last_class', '')
        focus_topic = self.diagnostic_answers.get('focus', '')
        
        query_parts = []
        if upcoming_test:
            query_parts.append(f"test preparation: {upcoming_test}")
        if last_class:
            query_parts.append(f"recent topic: {last_class}")
        if focus_topic:
            query_parts.append(f"focus area: {focus_topic}")
            
        query = " ".join(query_parts) if query_parts else "general math exercise"
        
        relevant_chunks = retrieve_relevant_chunks(query, self.pinecone_index)
        
        if not relevant_chunks:
            return False
            
        exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks))
        available_ids = [ex_id for ex_id in exercise_ids if ex_id not in self.recently_asked_exercise_ids]
        
        if not available_ids:
            available_ids = exercise_ids
            
        if not available_ids:
            return False
            
        chosen_id = random.choice(available_ids)
        self.current_exercise = self._get_exercise_by_id(chosen_id)
        
        if self.current_exercise:
            self.recently_asked_exercise_ids.append(chosen_id)
            if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
                self.recently_asked_exercise_ids.pop(0)
            
            # Generate SVG description if available
            if self.current_exercise.get("svg"):
                try:
                    svg_content = self.current_exercise["svg"][0]
                    self.current_svg_description = describe_svg_content(svg_content)
                except Exception as e:
                    logger.error(f"Error processing SVG for exercise {chosen_id}: {e}")
                    self.current_svg_description = "Image description unavailable."
            
            return True
            
        return False

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Get exercise by ID from the dataset."""
        return next((ex for ex in self.exercises_data if ex.get("canonical_exercise_id") == exercise_id), None)

    def _get_current_question(self) -> str:
        """Get the current question text with SVG handling."""
        if not (self.current_exercise and 
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list)):
            return "No question available."

        questions = self.current_exercise["text"]["question"]
        if not (0 <= self.current_question_index < len(questions)):
            return "No question available."

        q_text = questions[self.current_question_index]
        q_text = clean_math_text(q_text)
        
        # Generate SVG file if available and not already generated
        if self.current_exercise.get("svg") and not self.svg_generated_for_question:
            svg_reference = self._generate_and_save_svg()
            q_text += svg_reference
            self.svg_generated_for_question = True
        elif self.current_svg_file_path and self.svg_generated_for_question:
            # Reuse existing SVG file reference
            q_text += f"\n\n[Image File: {self.current_svg_file_path.as_posix()}]"
        
        if self.user_language == "en":
            return translate_text_to_english(q_text)
        return q_text

    def _generate_and_save_svg(self) -> str:
        """Generate and save SVG file. Returns the file reference text."""
        if not (self.current_exercise and self.current_exercise.get("svg")):
            return ""
            
        try:
            svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
            svg_content = self.current_exercise["svg"][svg_content_idx]
            
            if not svg_content:
                return ""
                
            # Generate unique filename
            svg_filename = f"exercise_{self.current_exercise['canonical_exercise_id']}_q{self.current_question_index}_{uuid.uuid4().hex[:8]}.svg"
            svg_filepath = SVG_OUTPUT_DIR / svg_filename
            
            try:
                with open(svg_filepath, "w", encoding="utf-8") as f:
                    f.write(svg_content)
                
                # Store the file path for reuse
                self.current_svg_file_path = svg_filepath
                
                file_reference = f"\n\n[Image File: {svg_filepath.as_posix()}]"
                
                # Add description if available
                if self.current_svg_description:
                    file_reference += f"\n[Image Description: {self.current_svg_description}]"
                    
                return file_reference
                
            except Exception as e:
                logger.error(f"Error saving SVG file: {e}")
                return "\n\n[Image: Error generating image file]"
                
        except Exception as e:
            logger.error(f"Error processing SVG for question: {e}")
            return ""

    def _get_current_solution(self) -> str:
        """Get the current solution text."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index]
            sol_text = clean_math_text(sol_text)
            
            if self.user_language == "en":
                return translate_text_to_english(sol_text)
            return sol_text
        return "No solution available."

    def _generate_hint_or_guidance(self, user_answer: str, question: str) -> str:
        """Generate hint or guidance using LLM if not available in dataset."""
        # First try to get hint from dataset
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list) and
            self.guidance_level < len(self.current_exercise["text"]["hint"])):
            
            hint_text = self.current_exercise["text"]["hint"][self.guidance_level]
            hint_text = clean_math_text(hint_text)
            if self.user_language == "en":
                hint_text = translate_text_to_english(hint_text)
            return hint_text
        
        # Generate using LLM if not in dataset
        try:
            guidance_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor providing guidance.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines based on guidance level {self.guidance_level}:
                - Level 0: "Try reading the question again carefully..."
                - Level 1: Ask a guiding question to help them think
                - Level 2: Ask another guiding question with more direction
                - Level 3: Provide a hint without giving the full answer
                - Level 4+: Provide the full solution
                
                Be encouraging and supportive. Don't give away the answer directly unless it's level 4+.
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Question: {question}\nStudent's Answer: {answer}\nGuidance Level: {level}")
            ])
            
            guidance_chain = guidance_prompt | llm
            response = guidance_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "answer": user_answer,
                "level": self.guidance_level
            })
            
            return clean_math_text(response.content.strip())
            
        except Exception as e:
            logger.error(f"Error generating guidance: {e}")
            return "Let me help you think through this step by step..."

    def _evaluate_answer(self, user_answer: str, question: str) -> bool:
        """Evaluate if the answer is correct."""
        try:
            evaluation_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor evaluating a student's answer.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Task: Determine if the student's answer is CORRECT or INCORRECT.
                Respond with only "CORRECT" or "INCORRECT" followed by a brief encouraging comment.
                """),
                ("user", "Question: {question}\nStudent Answer: {answer}\nCorrect Solution: {solution}")
            ])
            
            evaluation_chain = evaluation_prompt | llm
            solution = self._get_current_solution()
            
            response = evaluation_chain.invoke({
                "question": question,
                "answer": user_answer,
                "solution": solution
            })
            
            result = clean_math_text(response.content.strip())
            return result.upper().startswith("CORRECT")
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return False

    def _generate_lesson_summary(self) -> str:
        """Generate lesson summary using LLM."""
        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor providing a lesson summary.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Task: Create a brief 4-5 sentence summary of what was learned and practiced in this lesson.
                Include:
                - Key concepts covered
                - Student's progress
                - Encouragement
                - Mathmatical problems and solutions discussed
                - Mention any exercises completed but not in full detail since this is a summary part.
                
                Keep it positive and concise.
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Please provide a summary of this math tutoring session.")
            ])
            
            summary_chain = summary_prompt | llm
            response = summary_chain.invoke({
                "chat_history": self.chat_history[-10:]
            })
            
            summary = clean_math_text(response.content.strip())
            
            closing = "Great, that was an awesome lesson! I'll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you're a genius. Bye!"
            
            return f"{summary}\n\n{closing}"
            
        except KeyboardInterrupt:
            logger.error("User interrupted the summary generation")
            return "Great job today! You worked through some challenging problems and showed excellent progress. Keep practicing and see you next time. Remember, you're a genius! Bye!"
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Great job today! You worked through some challenging problems. Keep practicing and see you next time. Remember, you're a genius! Bye!"

    def transition(self, user_input: str) -> str:
        """Main state machine transition logic following the 4-state specification."""
        
        # Reset inactivity timer on user input
        if user_input.strip():
            self.inactivity_timer.reset()
            
        # Detect user language
        if user_input:
            detected_lang = detect_language(user_input)
            if detected_lang in ["he", "en"]:
                self.user_language = detected_lang
                
        # Add to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=clean_math_text(user_input)))

        # Handle inappropriate content
        if user_input and self._detect_inappropriate_content(user_input):
            response = "Sorry, but discussing these topics is not allowed during the lesson."
            self.chat_history.append(AIMessage(content=response))
            return response

        # Handle humor - EXACT specification: "Detect "haha", "lol", or laughter emojis 😂"
        if user_input and self._detect_humor(user_input):
            response = "Haha! 😄 That's funny! Let's get back to our math lesson."
            self.chat_history.append(AIMessage(content=response))
            return response

        # Handle general conversation requests in a natural way
        if user_input and self._is_general_conversation_request(user_input):
            response = self._handle_general_conversation(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response

        # Handle unclear or minimal responses
        if user_input and self._is_unclear_response(user_input):
            response = self._handle_unclear_response(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response

        # STATE MACHINE LOGIC
        if self.state == State.STATE_1_OPENING:
            return self._handle_opening_state(user_input)
            
        elif self.state == State.STATE_2_DIAGNOSTIC:
            return self._handle_diagnostic_state(user_input)
            
        elif self.state == State.STATE_3_LEARNING:
            return self._handle_learning_state(user_input)
            
        elif self.state == State.STATE_4_SUMMARY:
            return self._handle_summary_state(user_input)

        # Default fallback
        return "I'm not sure how to help with that. Let's continue with our lesson."

    def _handle_opening_state(self, user_input: str) -> str:
        """Handle STATE_1: Opening (Small Talk) - Fixed sequence."""
        
        # Check if user wants to skip to math at any point
        if user_input and self._wants_to_skip_to_math(user_input):
            response = self._handle_skip_to_math_request(user_input)
            self.state = State.STATE_2_DIAGNOSTIC
            self.chat_history.append(AIMessage(content=response))
            return response
        
        if self.opening_step == 0:
            # Step 1: "Hey hey, how are you?"
            self.opening_step = 1
            response = "Hey hey, how are you?"
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif self.opening_step == 1:
            # Step 2: Respond to "how are you" then ask about their day
            # BUT FIRST: Check if user wants to skip to math
            if self._wants_to_skip_to_math(user_input):
                response = self._handle_skip_to_math_request(user_input)
                self.state = State.STATE_2_DIAGNOSTIC
                self.chat_history.append(AIMessage(content=response))
                return response
            
            self.opening_step = 2
            # Generate a natural response to their "how are you" answer, then ask about their day
            response = self._generate_natural_response_to_how_are_you(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif self.opening_step == 2:
            # Step 3: Respond to their day info, then ask about hobbies
            # BUT FIRST: Check if user wants to skip to math
            if self._wants_to_skip_to_math(user_input):
                response = self._handle_skip_to_math_request(user_input)
                self.state = State.STATE_2_DIAGNOSTIC
                self.chat_history.append(AIMessage(content=response))
                return response
            
            self.user_day_info = user_input
            self.opening_step = 3
            response = self._generate_natural_response_to_day_info(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif self.opening_step == 3:
            # Step 4: Personalized follow-up based on hobby
            self.user_hobby = user_input
            self.opening_step = 4
            
            # Check if the "hobby" is actually a math request
            if self._wants_to_skip_to_math(user_input):
                response = self._handle_skip_to_math_request(user_input)
                self.state = State.STATE_2_DIAGNOSTIC
                self.chat_history.append(AIMessage(content=response))
                return response
            
            response = self._generate_personalized_followup(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif self.opening_step == 4:
            # Step 5: Humorous reaction and transition
            # BUT FIRST: Check if user wants to skip to math
            if self._wants_to_skip_to_math(user_input):
                response = self._handle_skip_to_math_request(user_input)
                self.state = State.STATE_2_DIAGNOSTIC
                self.chat_history.append(AIMessage(content=response))
                return response
            
            context = f"User's hobby: {self.user_hobby}, Day info: {self.user_day_info}"
            humorous_response = self._generate_humorous_reaction(context)
            
            # Transition to diagnostic
            self.state = State.STATE_2_DIAGNOSTIC
            self.chat_history.append(AIMessage(content=humorous_response))
            return humorous_response

    def _handle_diagnostic_state(self, user_input: str) -> str:
        """Handle STATE_2: Diagnostic - 3 questions with natural conversation."""
        
        # Check if user wants to chat more or has other requests - use both detection methods
        if user_input and (self._wants_to_chat_more(user_input) or self._is_general_conversation_request(user_input)):
            response = self._handle_chat_request_before_diagnostic(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # If this is the first call (no user input), ask the first question
        if not user_input and len(self.diagnostic_answers) == 0:
            response = "Do you have a test coming up?"
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Process user input and advance to next question
        if len(self.diagnostic_answers) == 0:
            # Store answer to question 1 and ask question 2
            self.diagnostic_answers['test'] = user_input
            response = "What did you cover in the last class?"
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif len(self.diagnostic_answers) == 1:
            # Store answer to question 2 and ask question 3
            self.diagnostic_answers['last_class'] = user_input
            response = "What would you like to work on today?"
            self.chat_history.append(AIMessage(content=response))
            return response
            
        elif len(self.diagnostic_answers) == 2:
            # Store answer to question 3 and transition to learning
            self.diagnostic_answers['focus'] = user_input
            
            # Pick first exercise based on diagnostic
            if self._pick_exercise_based_on_diagnostic():
                self.state = State.STATE_3_LEARNING
                self.exercise_counter = 0
                self.lesson_complete = False
                self.current_question_index = 0
                self.guidance_level = 0
                
                question = self._get_current_question()
                response = f"Great! Let's start with this exercise:\n\n{question}"
                self.chat_history.append(AIMessage(content=response))
                return response
            else:
                response = "I'm having trouble finding a suitable exercise. Let's try a general math problem."
                self.chat_history.append(AIMessage(content=response))
                return response

    def _handle_learning_state(self, user_input: str) -> str:
        """Handle STATE_3: Learning Stage - 2 exercises with guidance."""
        
        if not self.current_exercise:
            if not self._pick_exercise_based_on_diagnostic():
                self.state = State.STATE_4_SUMMARY
                return self._generate_lesson_summary()
        
        # Handle conversation requests and session ending requests
        if user_input and self._is_session_ending_request(user_input):
            response = self._handle_session_ending_request(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
            
        # Handle general conversation requests during learning
        if user_input and self._is_general_conversation_request(user_input):
            response = self._handle_learning_conversation_request(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Handle irrelevant questions
        irrelevant_keywords = [
            "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
            "music", "movie", "politics", "food", "travel", "holiday"
        ]
        if any(word in user_input.lower() for word in irrelevant_keywords):
            response = self._get_localized_text("irrelevant_msg")
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Check for solution requests only (give up, skip)
        solution_keywords = ["solution", "answer", "give up", "skip", "פתרון", "תשובה"]
        if any(keyword in user_input.lower() for keyword in solution_keywords):
            solution = self._get_current_solution()
            solution_prefix = self._get_localized_text("solution_prefix")
            response = f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise()}"
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # ALL OTHER RESPONSES (including "don't know", "help", etc.) go through 5-step progression
        # Evaluate the answer - treat help requests as wrong answers to trigger progression
        question = self._get_current_question()
        
        # Help requests should be treated as wrong answers to follow 5-step progression
        help_keywords = ["hint", "help", "stuck", "don't know", "clue", "tip", "רמז", "עזרה", "i don't know"]
        is_help_request = any(keyword in user_input.lower() for keyword in help_keywords)
        
        # Check if answer is correct (but help requests are always treated as wrong)
        is_correct = False if is_help_request else self._evaluate_answer(user_input, question)
        
        if is_correct:
            response = f"Excellent! That's correct! 🎉\n\n{self._move_to_next_exercise()}"
            self.chat_history.append(AIMessage(content=response))
            return response
        else:
            # IMPROVED: 2-3 attempt limit with conversational, human-like responses
            self.guidance_level += 1
            
            if self.guidance_level == 1:
                # First attempt - encouraging and supportive
                response = self._generate_first_attempt_encouragement(user_input, question)
            elif self.guidance_level == 2:
                # Second attempt - guiding question with personality
                response = self._generate_conversational_guidance(user_input, question)
            else:  # guidance_level >= 3
                # After 2-3 attempts - provide solution in a conversational, tutoring way
                response = self._generate_conversational_solution(user_input, question)
            
            self.chat_history.append(AIMessage(content=response))
            return response

    def _handle_summary_state(self, user_input: str) -> str:
        """Handle STATE_4: Summary - End of lesson with conversation support."""
        
        # Handle requests for more practice
        if user_input and self._wants_more_practice(user_input):
            response = self._handle_more_practice_request(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Handle general conversation requests in summary state
        if user_input and self._is_general_conversation_request(user_input):
            response = self._handle_summary_conversation(user_input)
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Handle goodbyes and session ending
        if user_input and self._is_goodbye(user_input):
            response = "Thank you for the great session! Keep practicing and see you next time. You're doing amazing! 👋 Bye!"
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Default: Generate summary
        summary = self._generate_lesson_summary()
        self.lesson_complete = True  # Mark lesson as complete after summary
        self.chat_history.append(AIMessage(content=summary))
        return summary

    def _generate_guiding_question_via_llm(self, user_input: str, question: str, question_number: int) -> str:
        """Generate guiding questions via LLM API call as per specification."""
        try:
            # Add timeout and retry logic
            guiding_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate guiding question #{question_number} to help the student.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - This is guiding question #{question_number} of 2
                - Ask a question that guides them toward the solution step by step
                - Don't give away the answer directly
                - Focus on the mathematical concept or method
                - Be very encouraging and supportive - they might be struggling
                - Keep it concise (1-2 sentences)
                - Start with encouraging phrases like "Let's break this down", "Think about this", "You're on the right track"
                
                Student's response: {user_input}
                Math problem: {question}
                """),
                ("user", f"Generate guiding question #{question_number}:")
            ])
            
            guiding_chain = guiding_prompt | llm
            response = guiding_chain.invoke({})
            
            return clean_math_text(response.content.strip())
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return f"🤔 Let me ask you this: What do you think the first step should be?"
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            # Return fallback based on question number
            if question_number == 1:
                return f"🤔 Let me ask you this: What do you think the first step should be?"
            else:
                return f"💡 Think about it: What operation would help you solve this equation?"

    def _get_hint_from_dataset_or_generate(self) -> str:
        """Get hint from dataset or generate via LLM API call as per specification."""
        # First try to get hint from dataset
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list) and
            len(self.current_exercise["text"]["hint"]) > 0):
            
            hint_text = self.current_exercise["text"]["hint"][0]  # Use first hint
            hint_text = clean_math_text(hint_text)
            if self.user_language == "en":
                return translate_text_to_english(hint_text)
            return hint_text
        
        # Generate via LLM API call if missing from dataset
        try:
            question = self._get_current_question()
            solution = self._get_current_solution()
            
            hint_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate a helpful hint for this math problem.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Provide a hint that guides toward the solution without giving it away
                - Focus on the mathematical concept or first step
                - Be encouraging and supportive
                - Keep it concise (1-2 sentences)
                
                Problem: {question}
                Solution (for reference): {solution}
                """),
                ("user", "Generate a helpful hint:")
            ])
            
            hint_chain = hint_prompt | llm
            response = hint_chain.invoke({})
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "Think about what mathematical operation or concept applies here."

    def _generate_guiding_question(self, user_answer: str, question: str, context: str = "") -> str:
        """Generate a guiding question to help student think through the problem."""
        try:
            guiding_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate a guiding question to help the student think through the problem step by step.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Ask a question that guides them toward the solution
                - Don't give away the answer directly
                - Focus on the mathematical concept or method
                - Be encouraging and supportive
                - Keep it concise (1-2 sentences)
                
                Example guiding questions:
                - "What operation should we use first?"
                - "Can you identify what type of equation this is?"
                - "What do you think the first step should be?"
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Problem: {question}\nStudent's Answer: {answer}\nContext: {context}\n\nGenerate a helpful guiding question:")
            ])
            
            guiding_chain = guiding_prompt | llm
            response = guiding_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "answer": user_answer,
                "context": context
            })
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            guiding_text = self._get_localized_text("guiding_question")
            return f"{guiding_text}What do you think the first step should be?"

    def _generate_similar_question(self, original_question: str) -> str:
        """Use LLM to rephrase an exercise into a similar question."""
        try:
            similar_q_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Task:
                - Rephrase the following math exercise into a SIMILAR but different question.
                - Keep the same difficulty level.
                - Do not provide the solution.
                - Only return the new question text."""),
                ("user", "{original_question}")
            ])
            
            similar_q_chain = similar_q_prompt | llm
            response = similar_q_chain.invoke({"original_question": original_question})
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating similar question: {e}")
            return original_question

    def _generate_progressive_hint(self, hint_level: int = 0) -> Optional[str]:
        """Generate progressive hints based on level."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list)):
            
            hints = self.current_exercise["text"]["hint"]
            if hint_level < len(hints):
                hint_text = hints[hint_level]
                hint_text = clean_math_text(hint_text)
                return translate_text_to_english(hint_text) if self.user_language == "en" else hint_text
        return None

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str = "") -> Dict[str, Any]:
        """Enhanced answer evaluation with progressive guidance system."""
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Math AI tutor evaluating a student's answer.
            
            Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
            
            Evaluation Guidelines:
            1. Determine if the answer is CORRECT or INCORRECT
            2. If INCORRECT, identify the specific mistake or misconception
            3. Provide encouragement regardless of correctness
            4. DO NOT reveal the correct answer
            5. Be supportive and educational
            
            Response Format:
            CORRECT: [brief encouraging comment]
            OR
            INCORRECT: [brief explanation of what went wrong without giving the answer]
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question}\nStudent Answer: {answer}\nContext: {context}\n\nEvaluate the answer:"),
        ])
        
        evaluation_chain = evaluation_prompt | llm
        try:
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "question": question,
                "answer": user_input,
                "context": context
            })
            
            evaluation_result = clean_math_text(eval_response.content.strip())
            is_correct = evaluation_result.lower().startswith("correct:")
            is_partial = evaluation_result.lower().startswith("partial:")
            
            return {
                "is_correct": is_correct,
                "is_partial": is_partial,
                "feedback": evaluation_result,
                "needs_guidance": not is_correct
            }
        except Exception as e:
            logger.error(f"Error in answer evaluation: {e}")
            return {
                "is_correct": False,
                "feedback": "I couldn't evaluate your answer right now.",
                "needs_guidance": True
            }

    def _provide_progressive_guidance(self, user_input: str, question: str, context: str = "", is_forced: bool = False) -> str:
        """Provide progressive guidance based on attempts and current guidance level."""
        lang_dict = I18N[self.user_language]
        
        if self.guidance_level == 0:  # Encouragement
            self.guidance_level = 1
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            return f"{guiding_prefix}{guiding_q}"
            
        elif self.guidance_level == 1:  # Second Guiding Question
            self.guidance_level = 2
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            return f"{guiding_prefix}{guiding_q}"

        elif self.guidance_level == 2:  # Hint
            if not is_forced and not self.attempt_tracker.can_provide_hint():
                return f"{lang_dict['encouragement']}{lang_dict['try_again']}"
            self.guidance_level = 3
            hint = self._generate_progressive_hint(0)
            if hint:
                hint_prefix = lang_dict["hint_prefix"]
                return f"{hint_prefix}{hint}"
            else:
                self.guidance_level = 4
                return self._get_current_solution()
                
        else:  # guidance_level >= 3, provide solution
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise()}"

    def _generate_first_attempt_encouragement(self, user_input: str, question: str) -> str:
        """Generate encouraging response for first wrong attempt."""
        try:
            encouragement_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly, encouraging math tutor. The student got the answer wrong on their first try.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Be very encouraging and supportive
                - Show understanding that math can be tricky
                - Maybe add a little humor to lighten the mood
                - Suggest they take another look at the problem
                - Keep it conversational and human-like, not robotic
                - Maximum 2-3 sentences
                
                Examples:
                "No worries! Math can be a bit tricky sometimes. Let's take another look at this together - what do you think the first step should be?"
                "Hey, close but not quite there yet! These problems can be sneaky. Want to give it another shot?"
                """),
                ("user", f"Student's incorrect answer: {user_input}\nMath problem: {question}")
            ])
            
            chain = encouragement_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "No worries! Math can be tricky sometimes. Let's take another look at this together - what do you think we should try next?"
        except Exception as e:
            logger.error(f"Error generating encouragement: {e}")
            return "No worries! Math can be tricky sometimes. Let's take another look at this together - what do you think we should try next?"

    def _generate_conversational_guidance(self, user_input: str, question: str) -> str:
        """Generate conversational guidance for second attempt."""
        try:
            guidance_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly math tutor. The student has tried twice and is still struggling.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Be encouraging but also provide more specific help
                - Ask a guiding question that points them toward the right approach
                - Be conversational and supportive, like talking to a friend
                - Show understanding that they're trying hard
                - Use a bit of humor if appropriate
                - Maximum 2-3 sentences
                
                Examples:
                "I can see you're really thinking about this! Let me ask you this - what type of math operation do you think we need here?"
                "You're putting in good effort! Here's something to consider - what's the first thing you notice about this equation?"
                """),
                ("user", f"Student's second incorrect answer: {user_input}\nMath problem: {question}")
            ])
            
            chain = guidance_prompt | llm
            response = chain.invoke({})
            return response.content.strip()
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            return "I can see you're really thinking about this! Let me ask you this - what type of math operation do you think we need here?"
        except Exception as e:
            logger.error(f"Error generating conversational guidance: {e}")
            return "I can see you're really thinking about this! Let me ask you this - what type of math operation do you think we need here?"

    def _generate_conversational_solution(self, user_input: str, question: str) -> str:
        """Generate conversational solution explanation after 2-3 attempts."""
        try:
            solution_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly math tutor. The student has tried 2-3 times and needs the solution explained in a conversational way.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Be understanding and supportive - they tried their best
                - Explain the solution like you're talking to a friend, not reading from a textbook
                - Use conversational language and maybe some humor
                - Walk them through the solution step by step in a friendly way
                - Encourage them that this is part of learning
                - End on a positive note
                - Be natural and human-like, not robotic
                
                Examples:
                "Alright, no worries! You gave it a good shot. Let me walk you through this step by step..."
                "Hey, that's totally fine - sometimes these problems are trickier than they look! Here's how we can solve this..."
                """),
                ("user", f"Student's attempts: {user_input}\nMath problem: {question}\nCorrect solution: {self._get_current_solution()}")
            ])
            
            chain = solution_prompt | llm
            response = chain.invoke({})
            
            # Add transition to next exercise
            conversational_solution = response.content.strip()
            return f"{conversational_solution}\n\n{self._move_to_next_exercise()}"
            
        except KeyboardInterrupt:
            logger.error("User interrupted the LLM call")
            solution = self._get_current_solution()
            return f"Alright, no worries! You gave it a good shot. Let me walk you through this step by step: {solution}\n\n{self._move_to_next_exercise()}"
        except Exception as e:
            logger.error(f"Error generating conversational solution: {e}")
            solution = self._get_current_solution()
            return f"Alright, no worries! You gave it a good shot. Let me walk you through this step by step: {solution}\n\n{self._move_to_next_exercise()}"

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate a doubt clearing response using RAG."""
        try:
            # Use RAG to find relevant context
            relevant_chunks = retrieve_relevant_chunks(user_question, self.pinecone_index)
            context = "\n".join([chunk.get("text", "") for chunk in relevant_chunks[:5]])
            
            doubt_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor helping to clear student doubts.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Answer the student's question clearly and thoroughly
                - Use the provided context to give accurate information
                - Explain concepts step by step
                - Be encouraging and supportive
                - If context is insufficient, provide general mathematical guidance
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Context: {context}\n\nStudent Question: {question}\n\nProvide a helpful explanation:")
            ])
            
            doubt_chain = doubt_prompt | llm
            response = doubt_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "context": context,
                "question": user_question
            })
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'm here to help with your question. Could you please rephrase it or provide more details?"

    def _move_to_next_exercise(self) -> str:
        """Move to the next exercise or end the lesson."""
        self.exercise_counter += 1
        
        if self.exercise_counter >= 2:
            self.state = State.STATE_4_SUMMARY
            return "We've completed our exercises for today. Let me give you a summary."
        else:
            # Reset for next exercise
            self.current_question_index = 0
            self.guidance_level = 0
            self.svg_generated_for_question = False
            self.current_svg_file_path = None
            self.current_svg_description = None
            
            if self._pick_exercise_based_on_diagnostic():
                question = self._get_current_question()
                return f"Great! Now let's try another exercise:\n\n{question}"
            else:
                self.state = State.STATE_4_SUMMARY
                return "We've completed our exercises for today. Let me give you a summary."

    # -----------------------------
    # Advanced Methods from Original (All Sophisticated Features)
    # -----------------------------
    def _generate_guiding_question(self, user_answer: str, question: str, context: str = "") -> str:
        """Generate a guiding question to help student think through the problem."""
        try:
            guiding_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate a guiding question to help the student think through the problem step by step.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Ask a question that guides them toward the solution
                - Don't give away the answer directly
                - Focus on the mathematical concept or method
                - Be encouraging and supportive
                - Keep it concise (1-2 sentences)
                
                Example guiding questions:
                - "What operation should we use first?"
                - "Can you identify what type of equation this is?"
                - "What do you think the first step should be?"
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Problem: {question}\nStudent's Answer: {answer}\nContext: {context}\n\nGenerate a helpful guiding question:")
            ])
            
            guiding_chain = guiding_prompt | llm
            response = guiding_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "answer": user_answer,
                "context": context
            })
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            guiding_text = self._get_localized_text("guiding_question", fallback="🤔 Let me ask you this: ")
            return f"{guiding_text}What do you think the first step should be?"
        
    def _generate_similar_question(self, original_question: str) -> str:
        """Use LLM to rephrase an exercise into a similar question."""
        try:
            similar_q_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Task:
                - Rephrase the following math exercise into a SIMILAR but different question.
                - Keep the same difficulty level.
                - Do not provide the solution.
                - Only return the new question text."""),
                ("user", "{original_question}")
            ])
            
            similar_q_chain = similar_q_prompt | llm
            response = similar_q_chain.invoke({"original_question": original_question})
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating similar question: {e}")
            return original_question  # fallback to original

    def _generate_progressive_hint(self, hint_level: int = 0) -> Optional[str]:
        """Generate progressive hints based on level."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list)):
            
            hints = self.current_exercise["text"]["hint"]
            if hint_level < len(hints):
                hint_text = hints[hint_level]
                hint_text = clean_math_text(hint_text)
                return translate_text_to_english(hint_text) if self.user_language == "en" else hint_text
        return None

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str = "") -> Dict[str, Any]:
        """Enhanced answer evaluation with progressive guidance system."""
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Math AI tutor evaluating a student's answer.
            
            Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
            
            Evaluation Guidelines:
            1. Determine if the answer is CORRECT or INCORRECT
            2. If INCORRECT, identify the specific mistake or misconception
            3. Provide encouragement regardless of correctness
            4. DO NOT reveal the correct answer
            5. Be supportive and educational
            
            Response Format:
            CORRECT: [brief encouraging comment]
            OR
            INCORRECT: [brief explanation of what went wrong without giving the answer]
            OR
            PARTIAL: [brief explanation of what's right and what needs work]
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question}\nStudent Answer: {answer}\nCorrect Solution: {solution}\nContext: {context}\n\nEvaluate the answer:"),
        ])
        
        evaluation_chain = evaluation_prompt | llm
        try:
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "question": question,
                "answer": user_input,
                "solution": solution,
                "context": context
            })
            
            evaluation_result = clean_math_text(eval_response.content.strip())
            is_correct = evaluation_result.lower().startswith("correct:")
            is_partial = evaluation_result.lower().startswith("partial:")
            
            return {
                "is_correct": is_correct,
                "is_partial": is_partial,
                "feedback": evaluation_result,
                "needs_guidance": not is_correct
            }
        except Exception as e:
            logger.error(f"Error in answer evaluation: {e}")
            return {
                "is_correct": False,
                "is_partial": False,
                "feedback": "I couldn't evaluate your answer right now.",
                "needs_guidance": True
            }

    def _provide_progressive_guidance(self, user_input: str, question: str, context: str = "", is_forced: bool = False) -> str:
        """Provide progressive guidance based on attempts and current guidance level."""
        lang_dict = I18N[self.user_language]
        
        if self.attempt_tracker.guidance_level == 0:  # Encouragement
            self.attempt_tracker.guidance_level = 1
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict.get("guiding_question", "🤔 Let me ask you this: ")
            return f"{guiding_prefix}{guiding_q}"
            
        elif self.attempt_tracker.guidance_level == 1:  # Second Guiding Question
            self.attempt_tracker.guidance_level = 2
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict.get("guiding_question", "🤔 Let me ask you this: ")
            return f"{guiding_prefix}{guiding_q}"

        elif self.attempt_tracker.guidance_level == 2:  # Hint
            if not is_forced and not self.attempt_tracker.can_provide_hint():
                encouragement = lang_dict.get("encouragement", "You're making progress — give it a try first!")
                try_again = lang_dict.get("try_again", "Can you try again? Think about your approach.")
                return f"{encouragement} {try_again}"
            self.attempt_tracker.guidance_level = 3
            hint = self._generate_progressive_hint(0)
            if hint:
                hint_prefix = lang_dict.get("hint_prefix", "💡 Hint: ")
                return f"{hint_prefix}{hint}"
            else:
                self.attempt_tracker.guidance_level = 4
                return self._get_current_solution()
                
        else:  # guidance_level >= 3, provide solution
            solution_prefix = lang_dict.get("solution_prefix", "✅ Solution: ")
            solution = self._get_current_solution()
            return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise()}"

    def _handle_hint_request(self, user_input: str) -> str:
        """Handle explicit hint requests."""
        self.attempt_tracker.has_requested_hint = True
        return self._provide_progressive_guidance(user_input, self._get_current_question(), is_forced=True)

    def _handle_solution_request(self, user_input: str) -> str:
        """Handle explicit solution requests."""
        self.attempt_tracker.has_requested_solution = True
        solution_prefix = self._get_localized_text("solution_prefix", fallback="✅ Solution: ")
        solution = self._get_current_solution()
        return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise()}"

    def _reset_attempt_tracking(self):
        """Reset attempt tracking for new question."""
        if hasattr(self, 'attempt_tracker'):
            self.attempt_tracker.reset()
        self.guidance_level = 0

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        """Pick a new exercise using RAG based on query."""
        relevant_chunks = retrieve_relevant_chunks(query, self.pinecone_index, grade, topic)
        
        if not relevant_chunks:
            return False
            
        exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks))
        available_ids = [ex_id for ex_id in exercise_ids if ex_id not in self.recently_asked_exercise_ids]
        
        if not available_ids:
            available_ids = exercise_ids
            
        if not available_ids:
            return False
            
        chosen_id = random.choice(available_ids)
        self.current_exercise = self._get_exercise_by_id(chosen_id)
        
        if self.current_exercise:
            self.recently_asked_exercise_ids.append(chosen_id)
            if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
                self.recently_asked_exercise_ids.pop(0)
            return True
            
        return False

    def _move_to_next_exercise_or_question(self) -> str:
        """Move to next question or exercise."""
        # Check if there are more questions in current exercise
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("question") and
            isinstance(self.current_exercise["text"]["question"], list)):
            
            total_questions = len(self.current_exercise["text"]["question"])
            if self.current_question_index + 1 < total_questions:
                # Move to next question in same exercise
                self.current_question_index += 1
                self._reset_attempt_tracking()
                self.svg_generated_for_question = False
                
                question = self._get_current_question()
                return f"Great! Now let's try the next part:\n\n{question}"
        
        # No more questions, move to next exercise
        return self._move_to_next_exercise()

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response for doubt clearing."""
        try:
            doubt_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor answering a student's question.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Answer the question clearly and educationally
                - Use examples if helpful
                - Be encouraging and supportive
                - Connect to previously learned concepts
                - Keep explanations clear and concise
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's question: {question}\n\nPlease provide a helpful explanation:")
            ])
            
            doubt_chain = doubt_prompt | llm
            response = doubt_chain.invoke({
                "chat_history": self.chat_history[-5:],
                "question": user_question
            })
            
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'm here to help with your math questions. Can you please rephrase your question?"

    # -----------------------------
    # Enhanced Chat History Management and Simple Chatbot Integration
    # -----------------------------
    def _manage_simple_chat(self, user_input: str) -> str:
        """Handle simple chat using the original simple chatbot logic."""
        # Add user message to history
        if user_input:
            self.chat_history.append(HumanMessage(content=user_input))

        # Keep only recent history for LLM context
        recent_history = self.chat_history[-6:]

        # Use the simple chain for basic conversation
        response = chain.invoke({
            "input": user_input,
            "chat_history": recent_history
        })

        # Get AI response content
        ai_response_content = response.content

        # Add AI response to history
        self.chat_history.append(AIMessage(content=ai_response_content))

        return ai_response_content

# -----------------------------
# MAIN (Enhanced with original simple chat integration)
# -----------------------------
def main():
    # --- 2. INITIALIZE CHAT HISTORY ---
    # We'll use a simple Python list to store the chat history.
    chat_history = []
    
    # Check if we have all required files for advanced mode
    if not PARSED_INPUT_FILE.exists():
        print("📚 Advanced mode unavailable. Running in simple chat mode.")
        print("A_GUY the math tutor is ready. Type 'exit' to end the chat.")
        
        # Simple chatbot mode (original functionality preserved)
        while True:
            # Get user input.
            user_input = input("Student: ")
            
            # Add a check to prevent empty messages from being sent to the LLM.
            if not user_input.strip():
                print("Please type a message.")
                continue

            if user_input.lower() == "exit":
                break

            # --- 6. MANAGE THE CHAT HISTORY ---
            # Append the new user message to the history.
            chat_history.append(HumanMessage(content=user_input))

            # We want to only send the last 5-6 messages to the LLM to maintain context.
            recent_history = chat_history[-6:]

            # --- 7. INVOKE THE LLM WITH CONTEXT ---
            # The 'invoke' method runs the chain, passing in the user's input and
            # the recent chat history.
            response = chain.invoke(
                {
                    "input": user_input,
                    "chat_history": recent_history
                }
            )

            # Get the AI's response content.
            ai_response_content = response.content

            # --- 8. UPDATE THE HISTORY WITH THE AI'S RESPONSE ---
            # Append the AI's response to the history for the next turn.
            chat_history.append(AIMessage(content=ai_response_content))

            # Print the AI's response.
            print(f"Assistant: {ai_response_content}")

        print("Chat session ended.")
        return

    # Advanced mode with sophisticated FSM
    try:
        exercises = load_json(PARSED_INPUT_FILE)
        pinecone_index = get_pinecone_index()
        print("🚀 Advanced 4-State Tutoring System activated!")
    except Exception as e:
        logger.error(f"❌ Error loading advanced features: {e}")
        print("⚠️ Falling back to simple chat mode...")
        
        # Fallback to simple mode
        print("A_GUY the math tutor is ready. Type 'exit' to end the chat.")
        
        while True:
            user_input = input("Student: ")
            
            if not user_input.strip():
                print("Please type a message.")
                continue

            if user_input.lower() == "exit":
                break

            chat_history.append(HumanMessage(content=user_input))
            recent_history = chat_history[-6:]

            response = chain.invoke({
                "input": user_input,
                "chat_history": recent_history
            })

            ai_response_content = response.content
            chat_history.append(AIMessage(content=ai_response_content))
            print(f"Assistant: {ai_response_content}")

        print("Chat session ended.")
        return

    # Initialize the sophisticated FSM
    fsm = MathTutorFSM(exercises, pinecone_index)

    # --- 5. START THE ENHANCED CHAT LOOP ---
    print("A_GUY the advanced math tutor is ready. Type 'exit' to end the chat.")
    
    # Initial transition to start the 4-state conversation
    initial_response = fsm.transition("")
    print(f"A_GUY: {initial_response}")

    while True:
        try:
            # Enhanced input handling with typing detection
            if hasattr(fsm, 'inactivity_timer'):
                fsm.inactivity_timer.mark_typing()
            
            # Robust input handling
            try:
                user_input = input("Student: ").strip()
            except EOFError:
                # Handle EOF (Ctrl+Z on Windows, Ctrl+D on Unix)
                print("\n👋 Bye!")
                if hasattr(fsm, 'inactivity_timer'):
                    fsm.inactivity_timer.stop()
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\n👋 Bye!")
                if hasattr(fsm, 'inactivity_timer'):
                    fsm.inactivity_timer.stop()
                break
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            if hasattr(fsm, 'inactivity_timer'):
                fsm.inactivity_timer.stop()
            break
            
        # Handle empty input
        if not user_input:
            # If lesson is complete, give a gentle reminder
            if hasattr(fsm, 'lesson_complete') and fsm.lesson_complete:
                print("The lesson is complete! You can type 'exit' to end, or start a new conversation.")
            else:
                print("Please type a message.")
            continue
            
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            if hasattr(fsm, 'inactivity_timer'):
                fsm.inactivity_timer.stop()
            break

        # Use sophisticated FSM or fallback to simple chat
        try:
            response = fsm.transition(user_input)
            print(f"A_GUY: {response}")
        except Exception as e:
            logger.error(f"Error in FSM: {e}")
            # Fallback to simple chat for this interaction
            response = fsm._manage_simple_chat(user_input)
            print(f"A_GUY: {response}")

if __name__ == "__main__":
    main()
