import os
import json
import random
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

# -----------------------------
# GenAI Chat Client (using LangChain)
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt templates
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Math AI tutor. 
    Respond in the same language as the user's question (Hebrew or English). 
    Analyze the 'Question' and 'Context' provided. 
    Context may be in English or Hebrew; translate relevant parts to match the question's language. 
    If 'topic' or 'mathematical_concept' is in Hebrew, translate to English for internal use. 
    Use all context information to answer math exercise questions. 
    If context lacks crucial information, state what is missing. 
    Do not invent information. 
    Use exact text from context for hints or solutions. 
    Use image descriptions for understanding if provided."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context: {context}\n\nQuestion: {input}"),
])
rag_chain = rag_prompt | llm

small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor starting a casual conversation. 
    Personality: Warm, encouraging, approachable, enthusiastic about math, professional yet casual. 
    Guidelines: 
    - Respond in the user's language (English or Hebrew).
    - Keep responses short (1-2 sentences).
    - Be friendly but not overly familiar.
    - Reference general school topics lightly.
    - Avoid personal details and immediate math talk."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
small_talk_chain = small_talk_prompt | llm

hebrew_small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """אתה מורה מתמטיקה ידידותי שמתחיל שיחה נינוחה עם תלמיד.
    אישיות: חם, מעודד, נגיש, נלהב ממתמטיקה, מקצועי אך קליל.
    הנחיות:
    - הגב בעברית.
    - שמור על תשובות קצרות (1-2 משפטים).
    - היה ידידותי אך לא מוכר מדי.
    - התייחס לנושאי בית ספר כלליים בקלילות.
    - הימנע מפרטים אישיים ואל תדבר מיד על מתמטיקה."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
hebrew_small_talk_chain = hebrew_small_talk_prompt | llm

personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor continuing a casual conversation. 
    Role: Continue naturally, show interest in their response, transition to academic topics. 
    Guidelines:
    - Respond in the user's language (English or Hebrew).
    - Keep responses short (1-2 sentences).
    - Acknowledge their response.
    - Reference school/learning generally.
    - Maintain a warm, encouraging tone."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
personal_followup_chain = personal_followup_prompt | llm

hebrew_personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """אתה מורה מתמטיקה ידידותי שממשיך שיחה נינוחה עם תלמיד.
    תפקיד: המשך באופן טבעי, הראה עניין בתגובתם, עבור בהדרגה לנושאים אקדמיים.
    הנחיות:
    - הגב בעברית.
    - שמור על תשובות קצרות (1-2 משפטים).
    - הכר בתגובתם.
    - התייחס לבית ספר/למידה באופן כללי.
    - שמור על טון חם ומעודד."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
hebrew_personal_followup_chain = hebrew_personal_followup_prompt | llm

# Localization
I18N = {
    "choose_language": "Choose language:\n1) English (default)\n2) Hebrew",
    "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
    "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? (e.g., {topics})",
    "ready_for_question": "Awesome! Let's start with the next exercise:",
    "hint_prefix": "💡 Hint: ",
    "solution_prefix": "✅ Solution: ",
    "wrong_answer": "Good attempt, but that's not correct. Would you like to try again or see a hint?",
    "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
    "no_more_hints": "No more hints available.",
    "no_relevant_exercises": "I couldn't find any relevant exercises for your query. Please try a different topic or question.",
    "ask_for_solution": "Would you like to see the solution?",
    "irrelevant_msg": "I can only help with math exercises and related questions.",
    "ask_for_doubts": "Great work! You've completed several exercises on {topic}. Do you have any questions or doubts about this topic?",
    "no_doubts_response": "Perfect! It looks like you understand {topic} well. Great job today!",
    "doubt_clearing_intro": "I'm here to help! Let me address your question about {topic}:",
    "ask_more_doubts": "Do you have any other questions or doubts about {topic}?",
    "session_ending": "Excellent work today! You've done well with {topic}. Keep practicing! This topic session is complete. See you next time! 👋",
    "doubt_answer_complete": "I hope that helps clarify things about {topic} for you!",
    "final_goodbye": "Thank you for the great {topic} session! Start a new session anytime for more math practice. Goodbye! 👋",
    "topic_session_complete": "🎉 Topic session for '{topic}' is complete! Start a new session with a different topic anytime.",
    "new_topic_offer": "Great! Let's work on a new topic. Which topic would you like to practice? (e.g., {topics})",
    "invalid_grade": "Please enter a valid grade (e.g., 7, 8).",
    "continue_topic": "Would you like to continue with more {topic} exercises or switch to a new topic?"
}

I18N_HEBREW = {
    "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)\n2) עברית",
    "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז, ח)",
    "ask_topic": "מעולה! כיתה {grade}. באיזה נושא תרצה להתאמן? (למשל, {topics})",
    "ready_for_question": "מצוין! בוא נתחיל עם התרגיל הבא:",
    "hint_prefix": "💡 רמז: ",
    "solution_prefix": "✅ פתרון: ",
    "wrong_answer": "ניסיון טוב, אבל זה לא נכון. רוצה לנסות שוב או לראות רמז?",
    "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
    "no_more_hints": "אין עוד רמזים זמינים.",
    "no_relevant_exercises": "לא מצאתי תרגילים רלוונטיים לשאלתך. נסה נושא או שאלה אחרים.",
    "ask_for_solution": "האם תרצה לראות את הפתרון?",
    "irrelevant_msg": "אני יכול לעזור רק עם תרגילי מתמטיקה ושאלות קשורות.",
    "ask_for_doubts": "עבודה נהדרת! השלמת מספר תרגילים בנושא {topic}. יש לך שאלות או ספקות בנושא זה?",
    "no_doubts_response": "מושלם! נראה שאתה מבין את {topic} היטב. עבודה מצוינת היום!",
    "doubt_clearing_intro": "אני כאן לעזור! הרשה לי לענות על שאלתך בנושא {topic}:",
    "ask_more_doubts": "יש לך שאלות או ספקות נוספים בנושא {topic}?",
    "session_ending": "עבודה מצוינת היום! הצלחת היטב עם תרגילי {topic}. המשך להתאמן! מפגש זה הושלם. להתראות! 👋",
    "doubt_answer_complete": "מקווה שזה מבהיר את הדברים בנושא {topic}!",
    "final_goodbye": "תודה על מפגש {topic} נהדר! התחל מפגש חדש בכל עת לתרגול נוסף. להתראות! 👋",
    "topic_session_complete": "🎉 מפגש הנושא '{topic}' הושלם! התחל מפגש חדש עם נושא אחר בכל עת.",
    "new_topic_offer": "נהדר! בוא נעבוד על נושא חדש. באיזה נושא תרצה להתאמן? (למשל, {topics})",
    #"invalid_grade": "אנא הזן כיתה תקפה (למ7, ).",
    "continue_topic": "האם תרצה להמשיך עם תרגילי {topic} נוספים או לעבור לנושא חדש?"
}

# FSM States
class State(Enum):
    START = auto()
    SMALL_TALK = auto()
    PERSONAL_FOLLOWUP = auto()
    ASK_GRADE = auto()
    EXERCISE_SELECTION = auto()
    QUESTION_ANSWER = auto()
    ASK_FOR_DOUBTS = auto()
    DOUBT_CLEARING = auto()
    SESSION_END = auto()
    END = auto()

# Helpers
def detect_user_language(text: str) -> str:
    """Detect if user input is in Hebrew or English."""
    if not text or not text.strip():
        return "en"
    hebrew_char_count = sum(1 for char in text if '\u0590' <= char <= '\u05FF')
    total_chars = len([char for char in text if char.isalpha()])
    return "he" if total_chars > 0 and hebrew_char_count / total_chars > 0.3 else "en"

def translate_to_language(text: str, target_language: str) -> str:
    """Translate text to target language using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a precise translator. Translate the following text to {target_language.capitalize()}. If it's already in {target_language.capitalize()}, return it as is. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error translating text to {target_language}: {str(e)}")
        return text

def translate_text_to_english(text: str) -> str:
    """Translate text to English using GenAI."""
    if not text or not text.strip():
        return text
    if not is_likely_hebrew(text):
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise translator. Translate the following text to English. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error translating text to English: {str(e)}")
        return text

def is_likely_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def get_localized_message(key: str, language: str, **kwargs) -> str:
    """Get localized message based on user language."""
    return I18N_HEBREW.get(key, I18N[key]).format(**kwargs) if language == "he" else I18N[key].format(**kwargs)

def load_json(p: Path):
    """Load JSON file."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {p}: {str(e)}")
        return []

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

def get_pinecone_index():
    """Initialize Pinecone index."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text."""
    if embedding_model is None:
        logger.error("Embedding model not loaded.")
        return []
    try:
        return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_chunks(query: str, pc_index: Any, grade: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from Pinecone."""
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []
    filter_dict = {k: {"$eq": v} for k, v in (("grade", grade), ("topic", topic)) if v}
    try:
        response = pc_index.query(vector=query_embedding, top_k=TOP_K_RETRIEVAL, include_metadata=True, filter=filter_dict or None)
        return [match.metadata for match in response.matches]
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}")
        return []

def describe_svg_content(svg_content: str) -> str:
    """Describe SVG content using GenAI."""
    try:
        svg_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code."),
            ("user", "Describe the following SVG content:\n{svg_input}"),
        ])
        svg_description_chain = svg_description_prompt | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return response.content
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."

# Dialogue FSM
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
        self.incorrect_attempts_count = 0
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 5
        self.small_talk_turns = 0
        self.user_language = "en"
        self.completed_exercises_count = 0
        self.doubt_questions_count = 0
        self.MAX_DOUBT_QUESTIONS = 2
        self.EXERCISES_BEFORE_DOUBT_CHECK = 4
        self.topic_exercises_count = 0

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        """Translate grade number to Hebrew."""
        grade_map = {"7": "ז", "8": "ח", "9": "ט", "10": "י"}
        return grade_map.get(grade_num, grade_num)

    def _generate_ai_small_talk(self, user_input: str = "") -> str:
        """Generate AI-based small talk response."""
        try:
            user_input = user_input.strip() or ("שלום" if self.user_language == "he" else "Hello")
            chain = hebrew_small_talk_chain if self.user_language == "he" else small_talk_chain
            response = chain.invoke({"chat_history": self.chat_history[-3:], "input": user_input})
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI small talk: {e}")
            return "שלום! איך הולך?" if self.user_language == "he" else "Hey! How's it going?"

    def _generate_ai_personal_followup(self, user_input: str = "") -> str:
        """Generate AI-based personal follow-up response."""
        try:
            user_input = user_input.strip() or ("זה נחמד" if self.user_language == "he" else "That's nice")
            chain = hebrew_personal_followup_chain if self.user_language == "he" else personal_followup_chain
            response = chain.invoke({"chat_history": self.chat_history[-3:], "input": user_input})
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI personal follow-up: {e}")
            return "נחמד! איזה נושא אהבת לאחרונה?" if self.user_language == "he" else "Nice! Which topic did you enjoy recently?"

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response to clear student's doubts using RAG."""
        try:
            user_lang = detect_user_language(user_question)
            translated_question = translate_text_to_english(user_question)
            retrieved_context = retrieve_relevant_chunks(
                translated_question, self.pinecone_index, self.hebrew_grade,
                None if self.topic and self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"] else self.topic
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            
            system_message = (
                """אתה מורה מתמטיקה מועיל שמתייחס לספק או שאלה של תלמיד.
                תפקיד: ספק הסברים ברורים ומפורטים, השתמש בהקשר, היה סבלני, פרק מושגים מורכבים, תגיב בעברית, הכר בחוסר מידע אם אין הקשר רלוונטי.
                הנחיות: התחל עם עידוד, תן הסבר שלב אחר שלב, השתמש בדוגמאות מההקשר, סיים עם שאלת אישור.""" if user_lang == "he" else
                """You are a helpful Math AI tutor addressing a student's doubt.
                Role: Provide clear explanations, use context, be patient, break down concepts, respond in English, acknowledge missing context.
                Guidelines: Start with encouragement, give step-by-step explanation, use context examples, end with confirmation question."""
            )
            
            doubt_clearing_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's Question: {question}\n\nRelevant Context: {context}"),
            ])
            doubt_clearing_chain = doubt_clearing_prompt | llm
            response = doubt_clearing_chain.invoke({
                "chat_history": self.chat_history[-5:], "question": user_question, "context": context_str
            })
            topic_name = self.topic or "this topic"
            intro_msg = get_localized_message('doubt_clearing_intro', user_lang, topic=topic_name)
            complete_msg = get_localized_message('doubt_answer_complete', user_lang, topic=topic_name)
            return f"{intro_msg}\n\n{response.content.strip()}\n\n{complete_msg}"
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return ("אשמח לעזור, אבל יש בעיה בעיבוד השאלה. נסה לשאול אחרת?" if detect_user_language(user_question) == "he" else
                    "I'd love to help, but I'm having trouble processing your question. Could you rephrase it?")

    def _reset_for_new_topic(self):
        """Reset counters and state for a new topic session."""
        self.topic = None
        self.topic_exercises_count = 0
        self.doubt_questions_count = 0
        self.current_exercise = None
        self.current_question_index = 0
        self.current_hint_index = 0
        self.incorrect_attempts_count = 0
        self.current_svg_description = None
        self.recently_asked_exercise_ids.clear()

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Get exercise by ID."""
        return next((ex for ex in self.exercises_data if ex.get("canonical_exercise_id") == exercise_id), None)

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        """Retrieve relevant exercises using RAG."""
        relevant_chunks = retrieve_relevant_chunks(query, self.pinecone_index, grade, topic)
        if not relevant_chunks:
            self.current_exercise = self.current_svg_description = None
            return
        all_exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks))
        available_exercise_ids = [ex_id for ex_id in all_exercise_ids if ex_id not in self.recently_asked_exercise_ids]
        if not available_exercise_ids:
            logger.info("All retrieved exercises recently asked. Clearing history.")
            self.recently_asked_exercise_ids.clear()
            available_exercise_ids = all_exercise_ids
        if not available_exercise_ids:
            self.current_exercise = self.current_svg_description = None
            return
        chosen_exercise_id = random.choice(available_exercise_ids)
        self.current_exercise = self._get_exercise_by_id(chosen_exercise_id)
        if not self.current_exercise:
            self.current_svg_description = None
            return
        self.current_hint_index = self.current_question_index = self.incorrect_attempts_count = 0
        self.recently_asked_exercise_ids.append(chosen_exercise_id)
        if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
            self.recently_asked_exercise_ids.pop(0)
        if self.current_exercise.get("svg"):
            try:
                self.current_svg_description = describe_svg_content(self.current_exercise["svg"][0])
            except Exception as e:
                logger.error(f"Error processing SVG for exercise {chosen_exercise_id}: {e}")
                self.current_svg_description = "Image description unavailable."

    def _validate_exercise_data(self) -> bool:
        """Validate current exercise data."""
        return (self.current_exercise and
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list) and
                0 <= self.current_question_index < len(self.current_exercise["text"]["question"]))

    def _get_current_question(self) -> str:
        """Get current question in user's language."""
        if not self._validate_exercise_data():
            logger.warning("Invalid exercise data structure for question retrieval.")
            return get_localized_message("no_relevant_exercises", self.user_language)
        q_text = self.current_exercise["text"]["question"][self.current_question_index].replace('$', '')
        if self.current_exercise.get("svg") and len(self.current_exercise["svg"]) > 0:
            try:
                svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
                svg_content = self.current_exercise["svg"][svg_content_idx]
                if svg_content:
                    svg_filename = f"exercise_{self.current_exercise['canonical_exercise_id']}_{uuid.uuid4().hex}.svg"
                    svg_filepath = SVG_OUTPUT_DIR / svg_filename
                    with open(svg_filepath, "w", encoding="utf-8") as f:
                        f.write(svg_content)
                    image_file_text = f"[Image File: {svg_filepath.as_posix()}]" if self.user_language == "en" else f"[קובץ תמונה: {svg_filepath.as_posix()}]"
                    q_text += f"\n\n{image_file_text}"
                    if self.current_svg_description:
                        desc_text = f"[Image Description: {self.current_svg_description}]"
                        if self.user_language == "he":
                            desc_text = f"[תיאור התמונה: {translate_to_language(self.current_svg_description, 'he')}]"
                        q_text += f"\n{desc_text}"
            except Exception as e:
                logger.error(f"Error processing SVG for question: {e}")
                q_text += f"\n\n{'[Image: Error generating image file]' if self.user_language == 'en' else '[תמונה: שגיאה ביצירת קובץ התמונה]'}"
        return q_text if self.user_language == "en" else translate_to_language(q_text, "he") if not is_likely_hebrew(q_text) else q_text

    def _get_current_solution(self) -> str:
        """Get current solution in user's language."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index].replace('$', '')
            return sol_text if self.user_language == "en" else translate_to_language(sol_text, "he") if not is_likely_hebrew(sol_text) else sol_text
        return "No solution available." if self.user_language == "en" else "אין פתרון זמין."

    def _get_current_hint(self) -> Optional[str]:
        """Get current hint in user's language."""
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list) and
            self.current_hint_index < len(self.current_exercise["text"]["hint"])):
            hint_text = self.current_exercise["text"]["hint"][self.current_hint_index].replace('$', '')
            self.current_hint_index += 1
            return hint_text if self.user_language == "en" else translate_to_language(hint_text, "he") if not is_likely_hebrew(hint_text) else hint_text
        return None

    def _move_to_next_exercise_or_question(self) -> str:
        """Move to next question or exercise, trigger doubt check if needed."""
        if self._validate_exercise_data() and self.current_question_index < len(self.current_exercise["text"]["question"]) - 1:
            self.current_question_index += 1
            self.current_hint_index = self.incorrect_attempts_count = 0
            next_question_msg = "Next question:" if self.user_language == "en" else "השאלה הבאה:"
            return f"\n\n{next_question_msg}\n{self._get_current_question()}"
        
        self.completed_exercises_count += 1
        self.topic_exercises_count += 1
        if self.topic_exercises_count >= self.EXERCISES_BEFORE_DOUBT_CHECK:
            self.state = State.ASK_FOR_DOUBTS
            self.current_exercise = self.current_svg_description = None
            self.current_question_index = self.current_hint_index = self.incorrect_attempts_count = 0
            topic_name = self.topic or "this topic"
            return f"\n\n{get_localized_message('ask_for_doubts', self.user_language, topic=topic_name)}"
        
        self.current_exercise = self.current_svg_description = None
        self.current_question_index = self.current_hint_index = self.incorrect_attempts_count = 0
        query = f"Next exercise for grade {self.hebrew_grade}"
        topic_for_query = None if self.topic and self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"] else self.topic
        if topic_for_query:
            query += f" on topic {self.topic}"
        self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)
        if not self.current_exercise:
            self.state = State.ASK_FOR_DOUBTS
            topic_name = self.topic or "this topic"
            return f"\n\n{get_localized_message('ask_for_doubts', self.user_language, topic=topic_name)}"
        next_exercise_msg = "Next exercise:" if self.user_language == "en" else "התרגיל הבא:"
        return f"\n\n{next_exercise_msg}\n{self._get_current_question()}"

    def transition(self, user_input: str) -> str:
        """Handle state transitions based on user input."""
        text_lower = user_input.strip().lower()
        if user_input.strip():
            detected_lang = detect_user_language(user_input)
            self.user_language = "he" if detected_lang == "he" or any(char in user_input for char in "אבגדהוזחטיכלמנסעפצקרשת") else "en"
        if user_input:
            self.chat_history.append(HumanMessage(content=user_input))

        # Define keywords
        new_topic_indicators = ["new topic", "change topic", "different topic", "נושא חדש", "שנה נושא", "נושא אחר"]
        no_doubt_indicators = ["no", "nope", "nothing", "all good", "everything clear", "לא", "אין", "הכל בסדר", "הכל ברור"]
        doubt_indicators = ["yes", "yeah", "question", "doubt", "confused", "don't understand", "כן", "שאלה", "ספק", "מבולבל", "לא מבין"]
        continue_indicators = ["continue", "more", "keep going", "next exercise", "המשך", "עוד", "תרגיל הבא"]

        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_turns = 1
            ai_response = self._generate_ai_small_talk(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.SMALL_TALK:
            self.state = State.PERSONAL_FOLLOWUP
            self.small_talk_turns += 1
            ai_response = self._generate_ai_personal_followup(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.ASK_GRADE
            response_text = get_localized_message("ask_grade", self.user_language)
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_GRADE:
            if not user_input.strip().isdigit() or int(user_input.strip()) not in range(7, 11):
                response_text = get_localized_message("invalid_grade", self.user_language)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            self.grade = user_input.strip()
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.state = State.EXERCISE_SELECTION
            available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
            topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
            response_text = get_localized_message("ask_topic", self.user_language, grade=self.grade, topics=topics_str)
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = user_input.strip()
            self.topic_exercises_count = self.doubt_questions_count = 0
            query = f"Find an exercise for grade {self.hebrew_grade} on topic {self.topic}"
            topic_for_picking = None if self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic", "כל נושא"] else self.topic
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_picking)
            if not self.current_exercise:
                logger.info(f"No exercises found for grade {self.hebrew_grade} and topic {self.topic}. Trying without topic filter.")
                self._pick_new_exercise_rag(query=f"Find an exercise for grade {self.hebrew_grade}", grade=self.hebrew_grade)
            if not self.current_exercise:
                self.state = State.EXERCISE_SELECTION
                response_text = (get_localized_message("no_exercises", self.user_language, grade=self.grade, topic=self.topic) + "\n" +
                                get_localized_message("no_relevant_exercises", self.user_language) + "\n\n" +
                                ("אנא בחר נושא אחר:" if self.user_language == "he" else "Please choose another topic:"))
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            self.state = State.QUESTION_ANSWER
            response_text = f"{get_localized_message('ready_for_question', self.user_language)}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.QUESTION_ANSWER:
            irrelevant_keywords = [
                "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
                "music", "movie", "politics", "food", "travel", "holiday",
                "מתכון", "עוגה", "סיפור", "בדיחה", "מזג אוויר", "שיר", "חדשות", "כדורגל",
                "מוזיקה", "סרט", "פוליטיקה", "אוכל", "טיול", "חופשה"
            ]
            if any(word in text_lower for word in irrelevant_keywords):
                response_text = get_localized_message("irrelevant_msg", self.user_language)
                response_text += f"\n\n{'בואו נתמקד בתרגיל הבא:' if self.user_language == 'he' else 'Let’s focus on the next exercise:'}\n"
                response_text += self._move_to_next_exercise_or_question()
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            hint_keywords = ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused",
                             "רמז", "עזרה", "עזור", "תקוע", "לא יודע", "לא בטוח", "מבולבל"]
            if (text_lower in ["hint", "רמז"] or
                any(keyword in text_lower for keyword in hint_keywords) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("תן" in text_lower and any(keyword in text_lower for keyword in ["רמז", "עזרה"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                hint = self._get_current_hint()
                response_text = (f"{get_localized_message('hint_prefix', self.user_language)} {hint}" if hint else
                                 f"{get_localized_message('hint_prefix', self.user_language)} {get_localized_message('no_more_hints', self.user_language)}")
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            solution_keywords = ["solution", "answer", "pass", "skip", "give up",
                                "פתרון", "תשובה", "דלג", "וותר", "אני מוותר"]
            if (text_lower in ["solution", "pass", "פתרון", "דלג"] or
                any(keyword in text_lower for keyword in solution_keywords) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("תן לי" in text_lower and any(keyword in text_lower for keyword in ["פתרון", "תשובה"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("הראה לי" in text_lower and any(keyword in text_lower for keyword in ["פתרון", "תשובה"]))):
                solution = self._get_current_solution()
                response_text = f"{get_localized_message('solution_prefix', self.user_language)} {solution}"
                response_text += self._move_to_next_exercise_or_question()
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            if any(indicator in text_lower for indicator in new_topic_indicators):
                self._reset_for_new_topic()
                self.state = State.EXERCISE_SELECTION
                available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                response_text = get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Evaluate answer
            current_question = self._get_current_question()
            current_solution = self._get_current_solution()
            retrieved_context = retrieve_relevant_chunks(
                f"Question: {current_question} User's Answer: {user_input}",
                self.pinecone_index, self.hebrew_grade,
                None if self.topic and self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"] else self.topic
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")]) + (
                f"\n\nImage Description: {self.current_svg_description}" if self.current_svg_description else "")
            system_msg = (
                """אתה מורה מתמטיקה AI. הערך את התשובה של המשתמש לנכונות על סמך השאלה, הפתרון וההקשר. 
                הגב רק עם 'נכון:' או 'לא נכון: <הסבר קצר של הטעות>'. 
                היה קצר והתמקד במתמטיקה. אם המשתמש שואל שאלה במקום תשובה, הגב 'לא נכון: זה נראה כמו שאלה, לא תשובה.'""" if self.user_language == "he" else
                """You are a Math AI tutor. Evaluate the user's answer for correctness based on the question, solution, and context. 
                Respond ONLY with 'CORRECT:' or 'INCORRECT: <brief explanation of the mistake>'. 
                Be concise and focus on the math. If the user asks a question instead of an answer, respond 'INCORRECT: This appears to be a question, not an answer.'"""
            )
            evaluation_prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Context: {context}\n\nQuestion: {question}\nCorrect Solution (for reference only, DO NOT reveal): {solution}\nUser's Answer: {user_answer}"),
            ])
            evaluation_chain = evaluation_prompt | llm
            try:
                eval_response = evaluation_chain.invoke({
                    "chat_history": self.chat_history[-4:], "context": context_str,
                    "question": current_question, "solution": "[REDACTED]", "user_answer": user_input
                })
                evaluation_result = eval_response.content.strip()
            except Exception as e:
                logger.error(f"LLM Evaluation Error: {e}")
                evaluation_result = ("לא נכון: סליחה, לא הצלחתי להעריך את התשובה שלך כרגע." if self.user_language == "he" else
                                     "INCORRECT: Sorry, I couldn't evaluate your answer right now.")

            if evaluation_result.lower().startswith(("correct:", "נכון:")):
                response_text = "Great job! That's correct!" if self.user_language == "en" else "עבודה נהדרת! זה נכון!"
                response_text += self._move_to_next_exercise_or_question()
            else:
                self.incorrect_attempts_count += 1
                response_text = f"{evaluation_result}\n\n{get_localized_message('wrong_answer', self.user_language)}"
                if self.incorrect_attempts_count >= 3:
                    response_text += f"\n\n{get_localized_message('ask_for_solution', self.user_language)}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_FOR_DOUBTS:
            topic_name = self.topic or "this topic"
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                response_text = get_localized_message("no_doubts_response", self.user_language, topic=topic_name)
                response_text += f"\n\n{get_localized_message('continue_topic', self.user_language, topic=topic_name)}"
                self.state = State.SESSION_END
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if any(indicator in text_lower for indicator in new_topic_indicators):
                self._reset_for_new_topic()
                self.state = State.EXERCISE_SELECTION
                available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                response_text = get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.doubt_questions_count = 1
                self.state = State.DOUBT_CLEARING
                response_text = self._generate_doubt_clearing_response(user_input)
                response_text += f"\n\n{get_localized_message('ask_more_doubts', self.user_language, topic=topic_name)}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            # Treat any other input as a doubt
            self.doubt_questions_count = 1
            self.state = State.DOUBT_CLEARING
            response_text = self._generate_doubt_clearing_response(user_input)
            response_text += f"\n\n{get_localized_message('ask_more_doubts', self.user_language, topic=topic_name)}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.DOUBT_CLEARING:
            topic_name = self.topic or "this topic"
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                response_text = get_localized_message("no_doubts_response", self.user_language, topic=topic_name)
                response_text += f"\n\n{get_localized_message('continue_topic', self.user_language, topic=topic_name)}"
                self.state = State.SESSION_END
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if any(indicator in text_lower for indicator in new_topic_indicators):
                self._reset_for_new_topic()
                self.state = State.EXERCISE_SELECTION
                available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                response_text = get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if any(indicator in text_lower for indicator in continue_indicators):
                self.state = State.QUESTION_ANSWER
                query = f"Next exercise for grade {self.hebrew_grade}"
                topic_for_query = None if self.topic and self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"] else self.topic
                if topic_for_query:
                    query += f" on topic {self.topic}"
                self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)
                if not self.current_exercise:
                    self.state = State.EXERCISE_SELECTION
                    available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                    topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                    response_text = get_localized_message("no_exercises", self.user_language, grade=self.grade, topic=self.topic) + "\n" + get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                response_text = f"{get_localized_message('ready_for_question', self.user_language)}\n{self._get_current_question()}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if self.doubt_questions_count < self.MAX_DOUBT_QUESTIONS:
                self.doubt_questions_count += 1
                response_text = self._generate_doubt_clearing_response(user_input)
                response_text += f"\n\n{get_localized_message('ask_more_doubts', self.user_language, topic=topic_name)}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            response_text = self._generate_doubt_clearing_response(user_input)
            response_text += f"\n\n{get_localized_message('session_ending', self.user_language, topic=topic_name)}"
            self.state = State.SESSION_END
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.SESSION_END:
            topic_name = self.topic or "this topic"
            if any(indicator in text_lower for indicator in new_topic_indicators):
                self._reset_for_new_topic()
                self.state = State.EXERCISE_SELECTION
                available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                response_text = get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            if any(indicator in text_lower for indicator in continue_indicators):
                self.state = State.QUESTION_ANSWER
                query = f"Next exercise for grade {self.hebrew_grade}"
                topic_for_query = None if self.topic and self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"] else self.topic
                if topic_for_query:
                    query += f" on topic {self.topic}"
                self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)
                if not self.current_exercise:
                    self.state = State.EXERCISE_SELECTION
                    available_topics = list(set(ex.get("topic", "Unknown") for ex in self.exercises_data if ex.get("grade") == self.hebrew_grade))
                    topics_str = ", ".join(available_topics[:5]) if self.user_language == "he" else ", ".join(translate_text_to_english(t) for t in available_topics[:5]) or ("כל נושא" if self.user_language == "he" else "Any topic")
                    response_text = get_localized_message("no_exercises", self.user_language, grade=self.grade, topic=self.topic) + "\n" + get_localized_message("new_topic_offer", self.user_language, topics=topics_str)
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                response_text = f"{get_localized_message('ready_for_question', self.user_language)}\n{self._get_current_question()}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            response_text = get_localized_message("final_goodbye", self.user_language, topic=topic_name)
            self.state = State.END
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.END:
            return "להתראות! 👋" if self.user_language == "he" else "Goodbye! 👋"

        default_msg = "אני לא בטוח איך להמשיך. הקלד 'יציאה' כדי לצאת." if self.user_language == "he" else "I'm not sure how to proceed. Type 'exit' to quit."
        self.chat_history.append(AIMessage(content=default_msg))
        return default_msg

def create_chatbot_session(user_language: str) -> Optional[DialogueFSM]:
    """Create a chatbot session."""
    try:
        exercises_data = load_json(PARSED_INPUT_FILE)
        pinecone_index = get_pinecone_index()
        fsm = DialogueFSM(exercises_data, pinecone_index)
        fsm.user_language = user_language
        return fsm
    except Exception as e:
        logger.error(f"Error creating chatbot session: {e}")
        return None

def main():
    """Main function for testing."""
    print("Welcome to Bilingual Math Tutor!")
    print(get_localized_message("choose_language", "en"))
    lang_choice = input("Choose language (1 or 2): ").strip()
    user_language = 'he' if lang_choice == '2' else 'en'
    print(f"Selected language: {'Hebrew' if user_language == 'he' else 'English'}")
    
    fsm = create_chatbot_session(user_language)
    if not fsm:
        print("❌ Error creating chatbot session")
        return

    initial_response = fsm.transition("")
    print(f"Tutor: {initial_response}")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit", "done", "יציאה"}:
                print("👋 Bye!")
                break
            response = fsm.transition(user_input)
            print(f"Tutor: {response}")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

if __name__ == "__main__":
    main()