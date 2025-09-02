# chatbot.py (AI-Based Small Talk with Doubt Clearing)
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

# Set up logging - Reduced to WARNING and ERROR for cleaner output
# Change to logging.INFO if you need to debug again
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

# Prompt template for the RAG chatbot (used for final response generation)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Math AI tutor. You MUST respond ONLY in English, regardless of the language of the input. Carefully analyze the user's 'Question' and the 'Context' provided. The 'Context' will be in English. The 'Question' might be in English or Hebrew, but your response must ALWAYS be in English. Use all information in the 'Context' to answer the user's questions about math exercises. If the context lacks crucial information, state specifically what is missing. Do not make assumptions or invent information. When providing hints or solutions, use the EXACT TEXT from the context provided. If an image description is given, use it for understanding."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context (in English): {context}\n\nQuestion (might be Hebrew/English): {input}"),
])
rag_chain = rag_prompt | llm

# Prompt template for AI-based small talk
small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor starting a casual conversation with a student. 
    
    Your personality:
    - Warm, encouraging, and approachable
    - Enthusiastic about helping with math
    - Casual but professional tone
    - Show genuine interest in the student
    
    Guidelines:
    - Keep responses short and conversational (1-2 sentences max)
    - Be friendly but don't be overly familiar
    - You can reference general school topics, but keep it light
    - Examples of good small talk: asking how they're doing, mentioning it's nice to see them, asking about their day
    - Avoid being too specific about personal details
    - Always respond in English
    - Don't immediately jump into math - this is just friendly chat
    
    Here are some example conversation starters you can draw inspiration from (but create your own):
    - "Hey, hey! How are you?"
    - "Did you watch the game yesterday?"
    - "Hey! How's everything today?"
    - "Hey! How's my dear Shila doing?"
    
    Generate a natural, friendly greeting or small talk response."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
small_talk_chain = small_talk_prompt | llm

# Prompt template for AI-based personal follow-up
personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor continuing a casual conversation with a student before moving to math topics.
    
    Your role:
    - Continue the friendly conversation naturally
    - Show interest in their response to your previous message
    - Keep it casual and brief (1-2 sentences)
    - Gradually transition toward academic topics, but still keep it conversational
    
    Guidelines:
    - Acknowledge what they said in a friendly way
    - Ask a follow-up question or make a casual comment
    - You can reference school, learning, or recent activities in general terms
    - Keep the tone warm and encouraging
    - Always respond in English
    
    Here are some example follow-up prompts you can draw inspiration from (but create your own based on the conversation):
    - "So, what do you think about Siena's victory yesterday?"
    - "Which topic did you enjoy the most recently?"
    - "Was your last lesson easy or challenging?"
    - "So, have you finished all the preparations for the Passover camp?"
    
    Generate a natural follow-up response that builds on the conversation."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
personal_followup_chain = personal_followup_prompt | llm

# -----------------------------
# Localization (English only for prompts)
# -----------------------------
I18N = {
    "choose_language": "Choose language:\n1) English (default)",
    # Keep some example prompts for reference, but they won't be used directly
    "small_talk_examples": [
        "Hey, hey! How are you?",
        "Did you watch the game yesterday?", 
        "Hey! How's everything today?",
        "Hey! How's my dear Shila doing?",
    ],
    "personal_followup_examples": [
        "So, what do you think about Siena's victory yesterday?",
        "Which topic did you enjoy the most recently?",
        "Was your last lesson easy or challenging?", 
        "So, have you finished all the preparations for the Passover camp?",
    ],
    "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
    "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? (e.g., {topics})",
    "ready_for_question": "Awesome! Let's start with the next exercise:",
    "hint_prefix": "💡 Hint: ",
    "solution_prefix": "✅ Solution: ",
    "wrong_answer": "Good attempt, but that's not correct. Would you like to try again or see a hint?",
    "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
    "no_more_hints": "No more hints available.",
    "no_relevant_exercises": "I couldn't find any relevant exercises for your query. Please try a different topic or question.",
    "ask_for_solution": "Would you like to see the solution to view?",
    "irrelevant_msg": "I can only help with math exercises and related questions.",
    # New entries for doubt clearing and session ending
    "ask_for_doubts": "Great work! You've completed several exercises on {topic}. Do you have any questions or doubts about this topic?",
    "no_doubts_response": "Perfect! It looks like you understand {topic} well. Great job today!",
    "doubt_clearing_intro": "I'm here to help! Let me address your question about {topic}:",
    "ask_more_doubts": "Do you have any other questions or doubts about {topic}?",
    "session_ending": "Excellent work today! You've done really well with {topic} exercises and I'm glad I could help clear up any questions. Keep practicing and you'll continue to improve! This topic session is now complete. See you next time! 👋",
    "doubt_answer_complete": "I hope that helps clarify things about {topic} for you!",
    "final_goodbye": "Thank you for the great {topic} session! Feel free to start a new session anytime for more math practice. Goodbye! 👋",
    "topic_session_complete": "🎉 Topic session for '{topic}' is now complete! You can start a new session with a different topic anytime."
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    START = auto()
    SMALL_TALK = auto()
    PERSONAL_FOLLOWUP = auto()
    ASK_GRADE = auto()
    EXERCISE_SELECTION = auto()
    QUESTION_ANSWER = auto()
    ASK_FOR_DOUBTS = auto()        # New state
    DOUBT_CLEARING = auto()        # New state
    SESSION_END = auto()           # New state
    END = auto()

# -----------------------------
# Helpers
# -----------------------------

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
# Dialogue FSM
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
        self.incorrect_attempts_count = 0
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 5
        self.small_talk_turns = 0
        
        # New attributes for doubt clearing functionality
        self.completed_exercises_count = 0      # Track completed exercises per topic
        self.doubt_questions_count = 0          # Track doubt questions asked
        self.MAX_DOUBT_QUESTIONS = 2            # Maximum doubt questions allowed
        self.EXERCISES_BEFORE_DOUBT_CHECK = 4   # Number of exercises before asking for doubts
        self.topic_exercises_count = 0          # Track exercises completed in current topic

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        grade_map = {"7": "ז", "8": "ח", "9": "ט", "10": "י"}
        return grade_map.get(grade_num, grade_num)

    def _generate_ai_small_talk(self, user_input: str = "") -> str:
        """Generate AI-based small talk response."""
        try:
            # Handle empty input for initial greeting
            if not user_input.strip():
                user_input = "Hello"
            
            response = small_talk_chain.invoke({
                "chat_history": self.chat_history[-3:],  # Keep recent context
                "input": user_input
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI small talk: {e}")
            # Fallback to example prompts if AI fails
            return random.choice(I18N["small_talk_examples"])

    def _generate_ai_personal_followup(self, user_input: str = "") -> str:
        """Generate AI-based personal follow-up response."""
        try:
            # Handle empty input
            if not user_input.strip():
                user_input = "That's nice"
                
            response = personal_followup_chain.invoke({
                "chat_history": self.chat_history[-3:],  # Keep recent context
                "input": user_input
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI personal follow-up: {e}")
            # Fallback to example prompts if AI fails
            return random.choice(I18N["personal_followup_examples"])

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response to clear student's doubts using RAG."""
        try:
            # Translate the question if it's in Hebrew
            translated_question = translate_text_to_english(user_question)
            
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
                ("system", """You are a helpful Math AI tutor addressing a student's doubt or question. 
                
                Your role:
                - Provide clear, detailed explanations
                - Use the context to give accurate information
                - Be patient and encouraging
                - Break down complex concepts into simple steps
                - Always respond in English
                - If the context doesn't contain relevant information, acknowledge this and provide general guidance
                
                Guidelines:
                - Start with encouragement ("Great question!")
                - Give a clear, step-by-step explanation
                - Use examples from the context when available
                - End with a confirmation question like "Does this make sense?" or "Is this clear?"
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's Question: {question}\n\nRelevant Context: {context}"),
            ])
            
            doubt_clearing_chain = doubt_clearing_prompt | llm
            response = doubt_clearing_chain.invoke({
                "chat_history": self.chat_history[-5:],  # Recent context
                "question": translated_question,
                "context": context_str
            })
            
            topic_name = self.topic or "this topic"
            return f"{I18N['doubt_clearing_intro'].format(topic=topic_name)}\n\n{response.content.strip()}\n\n{I18N['doubt_answer_complete'].format(topic=topic_name)}"
            
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'd be happy to help with your question, but I'm having trouble processing it right now. Could you try asking it in a different way?"

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
        self.recently_asked_exercise_ids.clear()  # Clear to allow fresh exercises for new topic

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
        self.incorrect_attempts_count = 0

        # Add to recently asked list
        self.recently_asked_exercise_ids.append(chosen_exercise_id)
        if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
            self.recently_asked_exercise_ids.pop(0)

        # Generate SVG description if available
        if self.current_exercise.get("svg"):
            try:
                svg_content = self.current_exercise["svg"][0]
                self.current_svg_description = describe_svg_content(svg_content)
            except Exception as e:
                logger.error(f"Error processing SVG for exercise {chosen_exercise_id}: {e}")
                self.current_svg_description = "Image description unavailable."

    def _get_current_question(self) -> str:
        """
        Retrieves the current question text for the exercise.
        Handles SVG content by saving it to a file and providing the path/description.
        """
        # 1. Validate exercise data and question index
        if not (self.current_exercise and 
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list)):
            logger.warning("Invalid exercise data structure for question retrieval.")
            return translate_text_to_english("No question available.")

        questions = self.current_exercise["text"]["question"]
        if not (0 <= self.current_question_index < len(questions)):
            logger.warning(f"Invalid question index: {self.current_question_index} for exercise with {len(questions)} questions.")
            return translate_text_to_english("No question available.")

        # 2. Get the specific question text for the current index
        q_text = questions[self.current_question_index]

        # 3. Handle SVG content if present for this exercise
        if self.current_exercise.get("svg") and len(self.current_exercise["svg"]) > 0:
            try:
                svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
                svg_content = self.current_exercise["svg"][svg_content_idx]
                
                if svg_content:
                    svg_filename = f"exercise_{self.current_exercise['canonical_exercise_id']}_{uuid.uuid4().hex}.svg"
                    svg_filepath = SVG_OUTPUT_DIR / svg_filename
                    try:
                        with open(svg_filepath, "w", encoding="utf-8") as f:
                            f.write(svg_content)
                        q_text += f"\n\n[Image File (for frontend display): {svg_filepath.as_posix()}]"
                    except Exception as e:
                        logger.error(f"Error saving SVG file: {e}")
                        q_text += "\n\n[Image: Error generating image file]"
                    
                    if self.current_svg_description:
                        translated_svg_desc = translate_text_to_english(self.current_svg_description)
                        q_text += f"\n[Image Description: {translated_svg_desc}]"
            except Exception as e:
                logger.error(f"Error processing SVG for question: {e}")

        # 4. Translate the final question text to English
        final_english_question = translate_text_to_english(q_text)
        return final_english_question

    def _get_current_solution(self) -> str:
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index]
            return translate_text_to_english(sol_text)
        return translate_text_to_english("No solution available.")

    def _get_current_hint(self) -> Optional[str]:
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list) and
            self.current_hint_index < len(self.current_exercise["text"]["hint"])):
            hint_text = self.current_exercise["text"]["hint"][self.current_hint_index]
            self.current_hint_index += 1
            return translate_text_to_english(hint_text)
        return None

    def _move_to_next_exercise_or_question(self) -> str:
        """Enhanced version that tracks completed exercises and triggers doubt checking per topic."""

        # 1. Check if there are more questions in the CURRENT exercise
        if (self.current_exercise and
            "text" in self.current_exercise and
            "question" in self.current_exercise["text"] and
            isinstance(self.current_exercise["text"]["question"], list) and
            self.current_question_index < len(self.current_exercise["text"]["question"]) - 1):
            
            # Move to next question within the same exercise
            self.current_question_index += 1
            self.current_hint_index = 0
            self.incorrect_attempts_count = 0
            return f"\n\nNext question:\n{self._get_current_question()}"

        else:
            # Current exercise is finished - increment counters
            self.completed_exercises_count += 1
            self.topic_exercises_count += 1
            
            # Check if we should ask for doubts after completing enough exercises for this topic
            if self.topic_exercises_count >= self.EXERCISES_BEFORE_DOUBT_CHECK:
                # Transition to doubt checking state for this topic
                self.state = State.ASK_FOR_DOUBTS
                self.current_exercise = None
                self.current_question_index = 0
                self.current_hint_index = 0
                self.incorrect_attempts_count = 0
                self.current_svg_description = None
                topic_name = self.topic or "this topic"
                return f"\n\n{I18N['ask_for_doubts'].format(topic=topic_name)}"
            
            # Otherwise, continue with normal exercise flow for the same topic
            # Clear state related to the old exercise
            self.current_exercise = None
            self.current_question_index = 0
            self.current_hint_index = 0
            self.incorrect_attempts_count = 0
            self.current_svg_description = None

            # Construct query for a new exercise in the same topic
            query = f"Next exercise for grade {self.hebrew_grade}"
            topic_for_query = None
            if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"]:
                query += f" on topic {self.topic}"
                topic_for_query = self.topic

            # Try to find a new exercise
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)

            # Handle case where no new exercise is found for this topic
            if not self.current_exercise:
                # No more exercises for this topic - trigger doubt checking
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{I18N['ask_for_doubts'].format(topic=topic_name)}"

            # Successfully found a new exercise
            return f"\n\nNext exercise:\n{self._get_current_question()}"

    def transition(self, user_input: str) -> str:
        text_lower = (user_input or "").strip().lower()

        # Add user input to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=user_input))

        # --- State Transitions ---
        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_turns = 1
            # Generate AI-based small talk
            ai_response = self._generate_ai_small_talk(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.SMALL_TALK:
            self.state = State.PERSONAL_FOLLOWUP
            self.small_talk_turns += 1
            # Generate AI-based personal follow-up
            ai_response = self._generate_ai_personal_followup(user_input)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.ASK_GRADE
            response_text = I18N["ask_grade"]
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_GRADE:
            self.grade = user_input.strip()
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.state = State.EXERCISE_SELECTION
            available_topics = list(set(
                ex.get("topic", "Unknown") for ex in self.exercises_data
                if ex.get("grade") == self.hebrew_grade
            ))
            topics_str = ", ".join(available_topics[:5]) if available_topics else "Any topic"
            response_text = I18N["ask_topic"].format(grade=self.grade, topics=topics_str)
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = user_input.strip()
            
            # Reset topic-specific counters when starting a new topic
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
                response_text = (I18N["no_exercises"].format(grade=self.grade, topic=self.topic) + "\n" +
                        I18N["no_relevant_exercises"] + "\n\nPlease choose another topic:")
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            self.state = State.QUESTION_ANSWER
            response_text = f"{I18N['ready_for_question']}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.QUESTION_ANSWER:
            # --- Irrelevant Question Handling ---
            irrelevant_keywords = [
                "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
                "music", "movie", "politics", "food", "travel", "holiday"
            ]
            if any(word in text_lower for word in irrelevant_keywords):
                response_text = "I can only help with math exercises and related questions."
                # Move to the next exercise/question
                response_text += "\n\nLet's focus on the next exercise:\n"
                response_text += self._move_to_next_exercise_or_question()
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
        
            # --- Handle Help Requests and Explicit Commands ---
            # Check for hint requests in various forms
            hint_keywords = ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused"]
            if (text_lower == "hint" or 
                any(keyword in text_lower for keyword in hint_keywords) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                
                hint = self._get_current_hint()
                if hint:
                    response_text = f"{I18N['hint_prefix']} {hint}"
                else:
                    response_text = f"{I18N['hint_prefix']} {I18N['no_more_hints']}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Check for solution requests in various forms
            solution_keywords = ["solution", "answer", "pass", "skip", "give up"]
            if (text_lower in {"solution", "pass"} or
                any(keyword in text_lower for keyword in solution_keywords) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer", "full solution"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("what is the" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("full solution" in text_lower) or
                ("correct answer" in text_lower)):
                
                solution = self._get_current_solution()
                response_text = f"{I18N['solution_prefix']} {solution}"
                response_text += self._move_to_next_exercise_or_question()
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # --- Handle Answer Evaluation ---
            else:
                current_question = self._get_current_question()
                current_solution = self._get_current_solution()

                # Use RAG to get context for evaluation
                retrieved_context = retrieve_relevant_chunks(
                    f"Question: {current_question} User's Answer: {user_input}",
                    self.pinecone_index,
                    grade=self.hebrew_grade,
                    topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
                )
                context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
                if self.current_svg_description:
                    context_str += f"\n\nImage Description: {self.current_svg_description}"

                # Evaluate answer using LLM - DO NOT reveal the correct answer in the feedback
                evaluation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a Math AI tutor. Evaluate the user's answer for correctness based on the question, solution, and context. Respond ONLY with 'CORRECT:' or 'INCORRECT: <brief explanation of the mistake or why it's wrong, without stating the correct value>'. Be concise and focus on the math. If the user is asking a question (e.g., 'what is the function?') instead of providing an answer, respond with 'INCORRECT: This appears to be a question, not an answer.'"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "Context: {context}\n\nQuestion: {question}\nCorrect Solution (for reference only, DO NOT reveal): {solution}\nUser's Answer: {user_answer}"),
                ])
                evaluation_chain = evaluation_prompt | llm
                try:
                    eval_response = evaluation_chain.invoke({
                        "chat_history": self.chat_history[-4:],
                        "context": context_str,
                        "question": current_question,
                        "solution": "[REDACTED]",
                        "user_answer": user_input
                    })
                    evaluation_result = eval_response.content.strip()
                except Exception as e:
                    logger.error(f"LLM Evaluation Error: {e}")
                    evaluation_result = "INCORRECT: Sorry, I couldn't evaluate your answer right now."

                # --- Process Evaluation Result ---
                if evaluation_result.lower().startswith("correct:"):
                    response_text = "✅ Correct!"
                    response_text += self._move_to_next_exercise_or_question()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                else: # Incorrect
                    self.incorrect_attempts_count += 1
                    # Provide feedback WITHOUT revealing the answer
                    response_text = f"{I18N['wrong_answer']}\nAI says: {evaluation_result}"

                    # Offer prompt for solution based on attempts
                    if self.incorrect_attempts_count <= 2:
                        response_text += f"\n\n{I18N['ask_for_solution']}"
                    else:
                        response_text += f"\n\n{I18N['ask_for_solution']}"

                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

        # NEW STATE: ASK_FOR_DOUBTS
        elif self.state == State.ASK_FOR_DOUBTS:
            # Check if user has doubts
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no doubts", "no questions", "i understand"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand"]
            
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                # Student has no doubts - move to session end for this topic
                self.state = State.SESSION_END
                response_text = I18N["no_doubts_response"].format(topic=topic_name)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                # Student has doubts - move to doubt clearing
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                
                # Extract the actual question if present
                if "?" in user_input:
                    doubt_response = self._generate_doubt_clearing_response(user_input)
                else:
                    doubt_response = f"I'm ready to help! What would you like me to explain or clarify about {topic_name}?"
                
                doubt_response += f"\n\n{I18N['ask_more_doubts'].format(topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            
            else:
                # Treat as a doubt/question
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                doubt_response += f"\n\n{I18N['ask_more_doubts'].format(topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        # NEW STATE: DOUBT_CLEARING
        elif self.state == State.DOUBT_CLEARING:
            # Check if user has more doubts or wants to end
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no more", "that's all", "thanks"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand"]
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                # No more doubts - end session for this topic
                self.state = State.SESSION_END
                response_text = I18N["session_ending"].format(topic=topic_name)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            elif self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                # Max doubts reached - end session for this topic
                doubt_response = self._generate_doubt_clearing_response(user_input)
                self.state = State.SESSION_END
                response_text = f"{doubt_response}\n\n{I18N['session_ending'].format(topic=topic_name)}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            else:
                # Handle another doubt question
                self.doubt_questions_count += 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                
                if self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                    # This was the last allowed doubt question for this topic
                    response_text = f"{doubt_response}\n\n{I18N['session_ending'].format(topic=topic_name)}"
                    self.state = State.SESSION_END
                else:
                    # Still can ask more doubts about this topic
                    response_text = f"{doubt_response}\n\n{I18N['ask_more_doubts'].format(topic=topic_name)}"
                
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

        # NEW STATE: SESSION_END (Per Topic)
        elif self.state == State.SESSION_END:
            # Topic session is complete - offer to start new topic or end completely
            topic_name = self.topic or "this topic"
            
            # Check if user wants to start a new topic
            new_topic_indicators = ["new topic", "another topic", "different topic", "next topic", "change topic"]
            if any(indicator in text_lower for indicator in new_topic_indicators):
                # Reset for new topic session
                self._reset_for_new_topic()
                self.state = State.EXERCISE_SELECTION
                
                # Get available topics for the grade
                available_topics = list(set(
                    ex.get("topic", "Unknown") for ex in self.exercises_data
                    if ex.get("grade") == self.hebrew_grade
                ))
                topics_str = ", ".join(available_topics[:5]) if available_topics else "Any topic"
                
                response_text = f"Great! Let's work on a new topic. Which topic would you like to practice? (e.g., {topics_str})"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            else:
                # Any other input ends the complete session
                response_text = I18N["final_goodbye"].format(topic=topic_name)
                self.state = State.END
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

        # --- End State ---
        elif self.state == State.END:
            return "👋 Goodbye!"

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
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            break

        response = fsm.transition(user_input)
        print(f"A_GUY: {response}")

if __name__ == "__main__":
    main()