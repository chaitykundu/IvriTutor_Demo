# chatbot.py (Enhanced AI-Based Math Tutor with Progressive Guidance and Improved SVG Handling)
import os
import json
import random
import time
import threading
from pathlib import Path
from enum import Enum, auto
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
import logging
import uuid

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
TYPING_DETECTION_THRESHOLD = 5  # Seconds to wait for complete input

# Progressive Guidance Settings
MIN_ATTEMPTS_BEFORE_HINT = 2
MIN_ATTEMPTS_BEFORE_SOLUTION = 3
MAX_GUIDANCE_LEVELS = 4  # 0=encouragement, 1=guiding_question, 2=hint, 3=solution

# -----------------------------
# GenAI Chat Client (using LangChain)
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt template for the RAG chatbot (bilingual support)
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

# Small talk prompt (bilingual)
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
    - Examples: "Hey! How are you doing today?", "Hi there! What's up?", "How's everything going?"
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
# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
        "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? (e.g., {topics})",
        "ready_for_question": "Awesome! Let's start with this exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "partial_answer": "Good start! Now, try to complete the final step.",  #updated
        "guiding_question": "🤔 Let me ask you this: ",
        "encouragement": "You’re making progress — give it try first!",
        "try_again": "Can you try again? Think about your approach.",
        "need_more_attempts": "Give it another try first - I believe you can work through this! {guiding_prompt}",
        "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
        "no_more_hints": "No more hints available. Would you like to see the solution?",
        "no_relevant_exercises": "I couldn't find any relevant exercises for your query.",
        "ask_for_solution": "Would you like me to show you the solution?",
        "irrelevant_msg": "I can only help with math exercises and related questions.",
        "inactivity_check": "Are you still there? I'm here to help whenever you're ready!",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "ask_for_doubts": "Great work! You've completed several exercises on {topic}. Do you have any questions or doubts about this topic?",
        "no_doubts_response": "Perfect! It looks like you understand {topic} well. Great job today!",
        "doubt_clearing_intro": "I'm here to help! Let me address your question about {topic}:",
        "ask_more_doubts": "Do you have any other questions about {topic}?",
        "doubt_answer_complete": "I hope that helps clarify things about {topic} for you!"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז, ח)",
        "ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? (לדוגמה: {topics})",
        "ready_for_question": "מעולה! בואו נתחיל עם התרגיל הזה:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "partial_answer": "פתיחה טובה! עכשיו נסה להשלים את הצעד הסופי.",  #updated
        "guiding_question": "🤔 תן לי לשאול אותך את זה: ",
        "encouragement": "אתה מתקדם - תנסה קודם!",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
        "need_more_attempts": "תן לזה עוד ניסיון - אני מאמין שאתה יכול לעבוד על זה!",
        "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
        "no_more_hints": "אין עוד רמזים זמינים. האם תרצה לראות את הפתרון?",
        "no_relevant_exercises": "לא הצלחתי למצוא תרגילים רלוונטיים לשאלתך.",
        "ask_for_solution": "האם תרצה שאראה לך את הפתרון?",
        "irrelevant_msg": "אני יכול לעזור רק עם תרגילי מתמטיקה ושאלות קשורות.",
        "inactivity_check": "אתה עדיין כאן? אני כאן לעזור בכל עת שתהיה מוכן!",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "ask_for_doubts": "עבודה מעולה! השלמת מספר תרגילים על {topic}. יש לך שאלות או ספקות על הנושא הזה?",
        "no_doubts_response": "מושלם! נראה שאתה מבין את {topic} היטב. עבודה נהדרת היום!",
        "doubt_clearing_intro": "אני כאן לעזור! תן לי לענות על השאלה שלך על {topic}:",
        "ask_more_doubts": "יש לך שאלות נוספות על {topic}?",
        "doubt_answer_complete": "אני מקווה שזה עוזר להבהיר דברים על {topic} עבורך!"
    }
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    START = auto()
    SMALL_TALK = auto()
    PERSONAL_FOLLOWUP = auto()
    ACADEMIC_TRANSITION = auto()
    ASK_GRADE = auto()
    EXERCISE_SELECTION = auto()
    QUESTION_ANSWER = auto()
    GUIDING_QUESTION = auto()
    PROVIDING_HINT = auto()
    ASK_FOR_DOUBTS = auto()
    DOUBT_CLEARING = auto()

# -----------------------------
# Helper Functions
# -----------------------------

def detect_language(text: str) -> str:
    """Detect if text is Hebrew or English."""
    if any('\u0590' <= char <= '\u05FF' for char in text):
        return "he"
    return "en"

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
        if is_likely_hebrew(text) and not is_likely_hebrew(translated):
             return translated
        elif not is_likely_hebrew(text):
             return text
        else:
             logger.warning(f"Potential translation issue. Input: {text}, Output: {translated}")
             return translated
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"[Translation Error: {text}]"

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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
        return response.content
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."

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
# Enhanced Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data, pinecone_index):
        self.state = State.START
        self.grade = None
        self.hebrew_grade = None
        self.topic = None
        self.exercises_data = exercises_data
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0
        self.pinecone_index = pinecone_index
        self.chat_history = []
        self.current_svg_description = None
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 0
        self.small_talk_turns = 0
        self.user_language = "en"
        
        self.small_talk_chain = small_talk_chain
        self.personal_followup_chain = personal_followup_chain  
        self.academic_transition_chain = academic_transition_chain

        # Enhanced attempt tracking
        self.attempt_tracker = AttemptTracker()
        
        # Doubt clearing functionality
        self.completed_exercises_count = 0
        self.doubt_questions_count = 0
        self.MAX_DOUBT_QUESTIONS = 2
        self.EXERCISES_BEFORE_DOUBT_CHECK = 3
        self.topic_exercises_count = 0
        
        # Enhanced inactivity timer
        self.inactivity_timer = EnhancedInactivityTimer(self._handle_inactivity)
        self._start_inactivity_timer()
        
        # SVG handling attributes
        self.current_svg_file_path = None  # Track the current SVG file
        self.svg_generated_for_question = False  # Track if SVG was generated for current question

    def _start_inactivity_timer(self):
        """Start or reset the inactivity timer."""
        self.inactivity_timer.reset()

    def _handle_inactivity(self):
        """Handle inactivity timeout - only triggers when truly inactive."""
        lang_dict = I18N[self.user_language]
        
        if self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            self._send_inactivity_message(lang_dict["inactivity_check"])
        else:
            self._send_inactivity_message(lang_dict["session_timeout"])
    
    def _send_inactivity_message(self, message):
        """Send inactivity message."""
        print(f"\n[INACTIVITY TIMEOUT] A_GUY: {message}")

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        grade_map = {"7": "ז", "8": "ח", "9": "ט", "10": "י"}
        return grade_map.get(grade_num, grade_num)

    def _get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        return text.format(**kwargs) if kwargs else text

    def _generate_ai_small_talk(self, user_input: str = "") -> str:
        """Generate AI-based small talk response."""
        try:
            response = self.small_talk_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI small talk: {e}")
            return "Hey! How's it going today?"

    def _generate_ai_personal_followup(self, user_input: str = "") -> str:
        """Generate AI-based personal follow-up response."""
        try:
            response = self.personal_followup_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI personal follow-up: {e}")
            return "That's interesting! How was your day yesterday?"

    def _generate_academic_transition(self, user_input: str = "") -> str:
        """Generate AI-based academic transition response."""
        try:
            response = self.academic_transition_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating academic transition: {e}")
            return "By the way, what have you been learning lately?"

    def _generate_guiding_question(self, user_answer: str, question: str, context: str) -> str:
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
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            guiding_text = self._get_localized_text("guiding_question")
            return f"{guiding_text}What do you think the first step should be?"

    def _generate_progressive_hint(self, hint_level: int = 0) -> Optional[str]:
        """Generate progressive hints based on level."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list)):
            
            hints = self.current_exercise["text"]["hint"]
            if hint_level < len(hints):
                hint_text = hints[hint_level].replace('$', '')
                return translate_text_to_english(hint_text) if self.user_language == "en" else hint_text
        return None

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str) -> Dict[str, Any]:
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
            
            evaluation_result = eval_response.content.strip()
            is_correct = evaluation_result.lower().startswith("correct:")
            is_partial = evaluation_result.lower().startswith("partial:") ##updated
            
            return {
                "is_correct": is_correct,
                "is_partial": is_partial,  #updated
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

    def _provide_progressive_guidance(self, user_input: str, question: str, context: str, is_forced: bool = False) -> str:
        """Provide progressive guidance based on attempts and current guidance level."""
        lang_dict = I18N[self.user_language]
        
        if self.attempt_tracker.guidance_level == 0:  # Encouragement
            
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            
            encouragement = lang_dict["encouragement"]
            try_again = lang_dict["try_again"]
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            
            # 🔄 UPDATED PART: rotate between encouragement and try_again
            if not hasattr(self, "last_msg_type"):
                self.last_msg_type = "encouragement"

            if self.last_msg_type == "encouragement":
                chosen_msg = try_again
                self.last_msg_type = "try_again"
            else:
                chosen_msg = encouragement
                self.last_msg_type = "encouragement"

            if is_forced:
                return f"{chosen_msg}\n\n{guiding_prefix}{guiding_q}"
            else:
                return f"{chosen_msg}\n\n{guiding_prefix}{guiding_q}"
            
        elif self.attempt_tracker.guidance_level == 1:  # Second Guiding Question
            self.attempt_tracker.guidance_level = 2
            self.state = State.GUIDING_QUESTION
            # Generate a different guiding question, possibly using more context or rephrasing
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            return f"{guiding_prefix}{guiding_q}"

        elif self.attempt_tracker.guidance_level == 2:  # Hint
            if not is_forced and not self.attempt_tracker.can_provide_hint():
                return f"{lang_dict['encouragement']}{lang_dict['try_again']}"
            self.attempt_tracker.guidance_level = 3
            self.state = State.PROVIDING_HINT
            hint = self._generate_progressive_hint(0)
            if hint:
                hint_prefix = lang_dict["hint_prefix"]
                return f"{hint_prefix}{hint}"
            else:
                self.attempt_tracker.guidance_level = 4
                return self._get_current_solution()
                
        else:  # guidance_level >= 3, provide solution
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise_or_question()}"

    def _handle_hint_request(self, user_input: str) -> str:
        """Handle explicit hint requests by providing guiding questions."""
        lang_dict = I18N[self.user_language]
        
        # Get current question context
        current_question = self._get_current_question()
        retrieved_context = retrieve_relevant_chunks(
            f"Question: {current_question} User's Answer: {user_input}",
            self.pinecone_index,
            grade=self.hebrew_grade,
            topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
        )
        context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
        if self.current_svg_description:
            context_str += f"\n\nImage Description: {self.current_svg_description}"
        
        # Instead of checking attempts, provide progressive guidance
        return self._provide_progressive_guidance(user_input, current_question, context_str, is_forced=True)

    def _handle_solution_request(self, user_input: str) -> str:
        """Handle explicit solution requests by providing guiding questions first."""
        lang_dict = I18N[self.user_language]
        
        # If they haven't gone through the guidance sequence, provide guiding questions first
        if self.attempt_tracker.guidance_level < 2:  # Less than hint level
            current_question = self._get_current_question()
            retrieved_context = retrieve_relevant_chunks(
                f"Question: {current_question} User's Answer: {user_input}",
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            if self.current_svg_description:
                context_str += f"\n\nImage Description: {self.current_svg_description}"
            
            # Provide guiding question instead of direct solution
            encouragement = lang_dict["encouragement"]
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            guiding_prefix = lang_dict["guiding_question"]
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            return f"{encouragement}{guiding_prefix}{guiding_q}"
        
        # If they've been through guidance, provide solution
        else:
            self.attempt_tracker.has_requested_solution = True
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            
            # Generate detailed explanation using AI
            try:
                explanation_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are a Math AI tutor providing a detailed solution explanation.
                    
                    Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                    
                    Guidelines:
                    - Explain the solution step by step
                    - Show the reasoning behind each step
                    - Help the student understand the concept
                    - Be clear and educational
                    - Keep it concise but thorough
                    - Reference the image/graph when relevant to explain concepts
                    """),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "Question: {question}\nSolution: {solution}\n\nProvide a step-by-step explanation:")
                ])
                
                explanation_chain = explanation_prompt | llm
                current_question = self._get_current_question()
                
                response = explanation_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "question": current_question,
                    "solution": solution
                })
                
                explanation = response.content.strip()
                
                # Generate NEW SVG for solution explanation
                svg_reference = ""
                if self.current_exercise and self.current_exercise.get("svg"):
                    svg_reference = self._generate_and_save_svg(for_solution_explanation=True)
                
                result = f"{solution_prefix}{solution}\n\n{explanation}"
                if svg_reference:
                    result += f"\n{svg_reference}"
                result += self._move_to_next_exercise_or_question()
                return result
                
            except Exception as e:
                logger.error(f"Error generating solution explanation: {e}")
                result = f"{solution_prefix}{solution}"
                result += self._move_to_next_exercise_or_question()
                return result

    def _reset_attempt_tracking(self):
        """Reset attempt tracking for new exercise."""
        self.attempt_tracker.reset()
        # Reset SVG tracking for new question/exercise
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        return next((ex for ex in self.exercises_data if ex.get("canonical_exercise_id") == exercise_id), None)

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        """Retrieve relevant exercises using RAG."""
        relevant_chunks = retrieve_relevant_chunks(query, self.pinecone_index, grade=grade, topic=topic)

        if not relevant_chunks:
            self.current_exercise = None
            self.current_svg_description = None
            return

        all_exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks))
        if not all_exercise_ids:
            self.current_exercise = None
            self.current_svg_description = None
            return

        # Filter out recently asked exercises
        available_exercise_ids = [ex_id for ex_id in all_exercise_ids if ex_id not in self.recently_asked_exercise_ids]

        if not available_exercise_ids:
            logger.info("All retrieved exercises were recently asked. Clearing history.")
            self.recently_asked_exercise_ids.clear()
            available_exercise_ids = all_exercise_ids

        if not available_exercise_ids:
            self.current_exercise = None
            self.current_svg_description = None
            return

        chosen_exercise_id = random.choice(available_exercise_ids)
        self.current_exercise = self._get_exercise_by_id(chosen_exercise_id)

        if not self.current_exercise:
            self.current_svg_description = None
            return

        # Reset exercise-specific counters
        self.current_hint_index = 0
        self.current_question_index = 0
        self.current_svg_description = None
        self._reset_attempt_tracking()
        
        # Reset SVG tracking for new exercise
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

        # Add to recently asked list
        self.recently_asked_exercise_ids.append(chosen_exercise_id)
        if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
            self.recently_asked_exercise_ids.pop(0)

        # Generate SVG description if available (but don't save file yet)
        if self.current_exercise.get("svg"):
            try:
                svg_content = self.current_exercise["svg"][0]
                self.current_svg_description = describe_svg_content(svg_content)
            except Exception as e:
                logger.error(f"Error processing SVG for exercise {chosen_exercise_id}: {e}")
                self.current_svg_description = "Image description unavailable."

    def _generate_and_save_svg(self, for_solution_explanation: bool = False) -> str:
        """Generate and save SVG file. Returns the file reference text."""
        if not (self.current_exercise and self.current_exercise.get("svg")):
            return ""
            
        try:
            svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
            svg_content = self.current_exercise["svg"][svg_content_idx]
            
            if not svg_content:
                return ""
                
            # Generate unique filename
            suffix = "_solution" if for_solution_explanation else ""
            svg_filename = f"exercise_{self.current_exercise['canonical_exercise_id']}_q{self.current_question_index}{suffix}_{uuid.uuid4().hex[:8]}.svg"
            svg_filepath = SVG_OUTPUT_DIR / svg_filename
            
            try:
                with open(svg_filepath, "w", encoding="utf-8") as f:
                    f.write(svg_content)
                
                # Store the file path for reuse
                if not for_solution_explanation:
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

    def _get_current_question(self) -> str:
        """Retrieves the current question text for the exercise."""
        if not (self.current_exercise and 
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list)):
            logger.warning("Invalid exercise data structure for question retrieval.")
            return "No question available."

        questions = self.current_exercise["text"]["question"]
        if not (0 <= self.current_question_index < len(questions)):
            logger.warning(f"Invalid question index: {self.current_question_index}")
            return "No question available."

        q_text = questions[self.current_question_index].replace(',', '')

        # Generate SVG only once per question (not for hints)
        if self.current_exercise.get("svg") and not self.svg_generated_for_question:
            svg_reference = self._generate_and_save_svg(for_solution_explanation=False)
            q_text += svg_reference
            self.svg_generated_for_question = True
        elif self.current_svg_file_path and self.svg_generated_for_question:
            # Reuse existing SVG file reference for hints
            q_text += f"\n\n[Image File: {self.current_svg_file_path.as_posix()}]"
            if self.current_svg_description:
                q_text += f"\n[Image Description: {self.current_svg_description}]"

        # Return in user's preferred language
        if self.user_language == "en":
            return translate_text_to_english(q_text)
        return q_text

    def _get_current_solution(self) -> str:
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index]
            sol_text = sol_text.replace(',', '')
            
            # Return in user's preferred language
            if self.user_language == "en":
                return translate_text_to_english(sol_text)
            return sol_text
        return "No solution available."

    def _move_to_next_exercise_or_question(self) -> str:
        """Enhanced version that tracks completed exercises and triggers doubt checking per topic."""
        
        # Check if there are more questions in the current exercise
        if (self.current_exercise and
            "text" in self.current_exercise and
            "question" in self.current_exercise["text"] and
            isinstance(self.current_exercise["text"]["question"], list) and
            self.current_question_index < len(self.current_exercise["text"]["question"]) - 1):
            
            # Move to next question within the same exercise
            self.current_question_index += 1
            self.current_hint_index = 0
            self._reset_attempt_tracking()  # This will reset SVG tracking too
            return f"\n\nNext question:\n{self._get_current_question()}"

        else:
            # Current exercise is finished - increment counters
            self.completed_exercises_count += 1
            self.topic_exercises_count += 1
            
            # Check if we should ask for doubts after completing enough exercises
            if self.topic_exercises_count >= self.EXERCISES_BEFORE_DOUBT_CHECK:
                self.state = State.ASK_FOR_DOUBTS
                self.current_exercise = None
                self.current_question_index = 0
                self.current_hint_index = 0
                self._reset_attempt_tracking()
                self.current_svg_description = None
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"
            
            # Continue with normal exercise flow
            self.current_exercise = None
            self.current_question_index = 0
            self.current_hint_index = 0
            self._reset_attempt_tracking()
            self.current_svg_description = None

            # Construct query for a new exercise
            query = f"Next exercise for grade {self.hebrew_grade}"
            topic_for_query = None
            if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"]:
                query += f" on topic {self.topic}"
                topic_for_query = self.topic

            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)

            if not self.current_exercise:
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"

            return f"\n\nNext exercise:\n{self._get_current_question()}"

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response to clear student's doubts using RAG."""
        try:
            # Translate the question if needed
            translated_question = translate_text_to_english(user_question) if self.user_language == "en" else user_question
            
            # Retrieve relevant context for the doubt
            retrieved_context = retrieve_relevant_chunks(
                translated_question,
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            
            # Create a doubt-clearing prompt
            doubt_clearing_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a helpful Math AI tutor addressing a student's doubt or question.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Provide clear, detailed explanations
                - Use the context to give accurate information
                - Be patient and encouraging
                - Break down complex concepts into simple steps
                - If context doesn't contain relevant information, acknowledge this
                - Start with encouragement
                - Give step-by-step explanation
                - Use examples from context when available
                - End with confirmation question
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's Question: {question}\n\nRelevant Context: {context}"),
            ])
            
            doubt_clearing_chain = doubt_clearing_prompt | llm
            response = doubt_clearing_chain.invoke({
                "chat_history": self.chat_history[-5:],
                "question": translated_question,
                "context": context_str
            })
            
            topic_name = self.topic or "this topic"
            intro = self._get_localized_text("doubt_clearing_intro", topic=topic_name)
            complete = self._get_localized_text("doubt_answer_complete", topic=topic_name)
            
            return f"{intro}\n\n{response.content.strip()}\n\n{complete}"
            
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'd be happy to help with your question, but I'm having trouble processing it right now. Could you try asking it in a different way?"

    def transition(self, user_input: str) -> str:
        """Enhanced FSM transition with progressive guidance and improved attempt tracking."""
        
        # Mark user activity (prevents premature inactivity timeout)
        if user_input.strip():
            self.inactivity_timer.reset()
            
        text_lower = (user_input or "").strip().lower()

        # Detect user language from input
        if user_input:
            detected_lang = detect_language(user_input)
            if detected_lang != self.user_language and detected_lang in ["he", "en"]:
                self.user_language = detected_lang
            
        # Add user input to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=user_input))

        # --- State Transitions ---
        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_turns = 1
            simple_greetings = ["Hey! How are you?", "Hi there!", "What's up?", "How's it going?"]
            ai_response = random.choice(simple_greetings)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.SMALL_TALK:
            self.state = State.PERSONAL_FOLLOWUP
            self.small_talk_turns += 1
            ai_response = self._generate_ai_personal_followup(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.ACADEMIC_TRANSITION
            ai_response = self._generate_academic_transition(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response
        
        elif self.state == State.ACADEMIC_TRANSITION:
            self.state = State.ASK_GRADE
            response_text = self._get_localized_text("ask_grade")
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_GRADE:
            self.grade = user_input.strip()
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.state = State.EXERCISE_SELECTION
            
            # Get available topics in Hebrew and translate if needed
            available_topics_hebrew = list(set(
                ex.get("topic", "Unknown") for ex in self.exercises_data
                if ex.get("grade") == self.hebrew_grade
            ))
            
            if available_topics_hebrew:
                if self.user_language == "en":
                    english_topics = [translate_text_to_english(topic) for topic in available_topics_hebrew[:]]
                    topics_str = ", ".join(english_topics)
                else:
                    topics_str = ", ".join(available_topics_hebrew[:])
            else:
                topics_str = "Any topic"
                
            response_text = self._get_localized_text("ask_topic", grade=self.grade, topics=topics_str)
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = user_input.strip()
            self.topic_exercises_count = 0
            self.doubt_questions_count = 0
            
            query = f"Find an exercise for grade {self.hebrew_grade} on topic {self.topic}"
            topic_for_picking = self.topic if self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_picking)

            if not self.current_exercise:
                logger.info(f"No exercises found for grade {self.hebrew_grade} and topic {self.topic}. Trying without topic filter.")
                query = f"Find an exercise for grade {self.hebrew_grade}"
                self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade)

            if not self.current_exercise:
                self.state = State.EXERCISE_SELECTION
                no_exercises = self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)
                no_relevant = self._get_localized_text("no_relevant_exercises")
                response_text = f"{no_exercises}\n{no_relevant}\n\nPlease choose another topic:"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            self.state = State.QUESTION_ANSWER
            ready_text = self._get_localized_text("ready_for_question")
            response_text = f"{ready_text}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            # Handle irrelevant questions
            irrelevant_keywords = [
                "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
                "music", "movie", "politics", "food", "travel", "holiday"
            ]
            if any(word in text_lower for word in irrelevant_keywords):
                irrelevant_msg = self._get_localized_text("irrelevant_msg")
                response_text = irrelevant_msg + "\n\nLet's focus on the current exercise."
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle explicit hint requests with attempt checking
            hint_keywords = ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused", "רמז", "עזרה"]
            if (text_lower == "hint" or 
                any(keyword in text_lower for keyword in hint_keywords) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                
                response_text = self._handle_hint_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle solution requests with attempt checking
            solution_keywords = ["solution", "answer", "pass", "skip", "give up", "show me the solution", "פתרון", "תשובה"]
            if (text_lower in {"solution", "pass"} or
                any(keyword in text_lower for keyword in solution_keywords) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"]))):
                
                response_text = self._handle_solution_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle answer evaluation with enhanced attempt tracking
            else:
                current_question = self._get_current_question()
                current_solution = self._get_current_solution()

                # Get context for evaluation
                retrieved_context = retrieve_relevant_chunks(
                    f"Question: {current_question} User's Answer: {user_input}",
                    self.pinecone_index,
                    grade=self.hebrew_grade,
                    topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
                )
                context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
                if self.current_svg_description:
                    context_str += f"\n\nImage Description: {self.current_svg_description}"

                # Evaluate answer
                evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, context_str)
                
                # Record the attempt
                should_offer_guidance = self.attempt_tracker.record_attempt(evaluation_result["is_correct"])
                
                if evaluation_result["is_correct"]:
                    response_text = "✅ Correct!"
                    response_text += self._move_to_next_exercise_or_question()
                    self.state = State.QUESTION_ANSWER
                    self._reset_attempt_tracking()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                #updated for partial answers
                elif evaluation_result["is_partial"]:  
                    encouragement = self._get_localized_text("encouragement")
                    partial_msg = self._get_localized_text("partial_answer")  # new string
                    response_text = f"{encouragement} {partial_msg}"
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                else:
                    # Incorrect answer - handle based on attempt count
                    feedback_lines = evaluation_result.get("feedback", self._get_localized_text("wrong_answer")).split('\n')
                    main_feedback = feedback_lines[0] if feedback_lines else self._get_localized_text("wrong_answer")
                    
                    if should_offer_guidance:
                        # Provide progressive guidance
                        guidance = self._provide_progressive_guidance(user_input, current_question, context_str)
                        response_text = f"{main_feedback}\n\n{guidance}"
                    else:
                        # Encourage another attempt
                        encouragement = self._get_localized_text("encouragement")
                        try_again = self._get_localized_text("try_again")
                        response_text = f"{main_feedback}\n\n{encouragement}{try_again}"
                    
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

        elif self.state == State.ASK_FOR_DOUBTS:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no doubts", "no questions", "לא", "אין", "בסדר"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה", "ספק"]
            
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                self.state = State.EXERCISE_SELECTION
                response_text = self._get_localized_text("no_doubts_response", topic=topic_name)
                response_text += "\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                
                if "?" in user_input:
                    doubt_response = self._generate_doubt_clearing_response(user_input)
                else:
                    doubt_response = f"I'm ready to help! What would you like me to explain or clarify about {topic_name}?"
                
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            
            else:
                # Treat as a doubt/question
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        elif self.state == State.DOUBT_CLEARING:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no more", "that's all", "thanks", "לא", "אין", "תודה", "זה הכל"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators) or self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                self.state = State.EXERCISE_SELECTION
                response_text = f"Perfect! Would you like to continue with more {topic_name} exercises or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
                
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.doubt_questions_count += 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                if self.doubt_questions_count < self.MAX_DOUBT_QUESTIONS:
                    doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            else:
                # Unclear response
                doubt_response = f"Could you clarify your question about {topic_name} or say 'no' if you're ready to move on?"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        # Default fallback
        response_text = "I'm not sure how to proceed. Type 'exit' to quit."
        self.chat_history.append(AIMessage(content=response_text))
        return response_text

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
    except Exception as e:
        logger.error(f"❌ Error loading data or connecting to Pinecone: {e}")
        return

    fsm = DialogueFSM(exercises, pinecone_index)

    # Initial transition to start the conversation
    initial_response = fsm.transition("")
    print(f"A_GUY: {initial_response}")

    while True:
        try:
            # Enhanced input handling with typing detection
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