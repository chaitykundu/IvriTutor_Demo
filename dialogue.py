import os
import json
import random
import time
import threading
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
import logging
import uuid
import prompt  # Assuming prompt module is available
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Inactivity and Interaction Settings
INACTIVITY_TIMEOUT = 60  # Seconds as per spec
TYPING_DETECTION_THRESHOLD = 5  # Seconds to wait for complete input
INAPPROPRIATE_BLACKLIST = ["violence", "politics", "drugs", "sex", "hate", "terrorism"]
HUMOR_INDICATORS = ["haha", "lol", "😂", "🤣", "funny"]
MAX_EXERCISES = 2  # Limit to 2 exercises for PoC
SUPPORTED_GRADES = ["7", "8"]  # Extensible list of supported grades

# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7 or 8)",
        "invalid_grade": "Sorry, I only support grades 7 and 8 for now. Please choose one (e.g., 7 or 8).",
        "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? Here are some suggestions: {topics}",
        "invalid_topic": "I couldn't find exercises for '{topic}'. Try one of these: {topics}. What would you like to work on?",
        "invalid_input": "Hmm, I didn't quite catch that! Could you share a bit more clearly? {retry_prompt}",
        "ready_for_question": "Awesome! Let's start with this exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "reread_question": "Try reading the question again carefully…",
        "encouragement": "You're on the right track! ",
        "try_again": "Can you try again? Think about your approach.",
        "need_more_attempts": "Give it another try first - I believe you can work through this!",
        "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
        "no_more_hints": "No more hints available. Would you like to see the solution?",
        "no_relevant_exercises": "I couldn't find any relevant exercises for your query.",
        "ask_for_solution": "Would you like me to show you the solution?",
        "inappropriate_msg": "Sorry, but discussing these topics is not allowed during the lesson.",
        "inactivity_check": "Hey, are you still there?",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "closing_message": "Great, that was an awesome lesson! I’ll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you’re a genius. Bye!",
        "continue_learning": "Awesome, let's keep going! Here's another exercise:"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז או ח)",
        "invalid_grade": "מצטער, אני תומך רק בכיתות ז ו-ח כרגע. אנא בחר אחת (למשל, ז או ח).",
        "ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? הנה כמה הצעות: {topics}",
        "invalid_topic": "לא מצאתי תרגילים עבור '{topic}'. נסה אחד מאלה: {topics}. על מה תרצה לעבוד?",
        "invalid_input": "הממ, לא ממש הבנתי! תוכל לשתף קצת יותר ברור? {retry_prompt}",
        "ready_for_question": "מעולה! בואו נתחיל עם התרגיל הזה:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "reread_question": "נסה לקרוא את השאלה שוב בקפידה…",
        "encouragement": "אתה בכיוון הנכון! ",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
        "need_more_attempts": "תן לזה עוד ניסיון - אני מאמין שאתה יכול לעבוד על זה!",
        "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
        "no_more_hints": "אין עוד רמזים זמינים. האם תרצה לראות את הפתרון?",
        "no_relevant_exercises": "לא הצלחתי למצוא תרגילים רלוונטיים לשאלתך.",
        "ask_for_solution": "האם תרצה שאראה לך את הפתרון?",
        "inappropriate_msg": "מצטער, אבל דיון בנושאים אלה אסור במהלך השיעור.",
        "inactivity_check": "היי, אתה עדיין שם?",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "closing_message": "שיעור נהדר! אשלח לך תרגילים דומים לתרגול וניפגש בשיעור הבא. אם יש לך שאלות, תרגיש חופשי לשלוח לי הודעה. ואם אתה נתקע - זכור, אתה גאון. ביי!",
        "continue_learning": "מעולה, בוא נמשיך! הנה תרגיל נוסף:"
    }
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    OPENING = auto()  # STATE_1: Small Talk
    DIAGNOSTIC = auto()  # STATE_2
    LEARNING = auto()  # STATE_3: 2 exercises
    SUMMARY = auto()  # STATE_4
    REREAD_QUESTION = auto()  # Guidance sub-states
    GUIDING_QUESTION_1 = auto()
    GUIDING_QUESTION_2 = auto()
    PROVIDING_HINT = auto()

# -----------------------------
# Enhanced Inactivity Timer
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
        self.typing_detected = True
        self.last_activity_time = time.time()
    
    def _check_inactivity(self):
        current_time = time.time()
        time_since_activity = current_time - self.last_activity_time
        if self.typing_detected and time_since_activity < TYPING_DETECTION_THRESHOLD:
            self.start()
        elif time_since_activity >= self.timeout:
            self.callback()
        else:
            remaining_time = self.timeout - time_since_activity
            self.timer = threading.Timer(remaining_time, self._check_inactivity)
            self.timer.daemon = True
            self.timer.start()

# -----------------------------
# Enhanced Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data, pinecone_index, llm):
        self.state = State.OPENING
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
        self.exercise_count = 0
        self.guidance_step = 0  # 0: initial, 1: reread, 2: GQ1, 3: GQ2, 4: hint, 5: solution
        self.opening_turn = 0  # Track small talk sub-turns
        self.diagnostic_turn = 0
        self.llm = llm
        self.user_language = "en"
        self.svg_generated_for_question = False
        self.current_svg_file_path = None
        
        # Initialize prompt chains
        self.rag_chain = prompt.get_rag_prompt(self.user_language) | llm
        self.small_talk_chain = prompt.get_small_talk_prompt(self.user_language) | llm
        self.personal_followup_chain = prompt.get_personal_followup_prompt(self.user_language) | llm
        self.academic_transition_chain = prompt.get_academic_transition_prompt(self.user_language) | llm
        self.personalized_followup_chain = prompt.get_personalized_followup_prompt(self.user_language) | llm
        self.humorous_reaction_chain = prompt.get_humorous_reaction_prompt(self.user_language) | llm
        
        self.inactivity_timer = EnhancedInactivityTimer(self._handle_inactivity)
        self._start_inactivity_timer()

    def _start_inactivity_timer(self):
        self.inactivity_timer.reset()

    def _handle_inactivity(self):
        lang_dict = I18N[self.user_language]
        if self.state in [State.LEARNING, State.REREAD_QUESTION, State.GUIDING_QUESTION_1, State.GUIDING_QUESTION_2, State.PROVIDING_HINT]:
            self._send_inactivity_message(lang_dict["inactivity_check"])
        else:
            self._send_inactivity_message(lang_dict["session_timeout"])
    
    def _send_inactivity_message(self, message):
        print(f"\n[INACTIVITY TIMEOUT] A_GUY: {message}")

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        grade_map = {"7": "ז", "8": "ח", "9": "ט", "10": "י"}
        return grade_map.get(grade_num, grade_num)

    def _get_localized_text(self, key: str, **kwargs) -> str:
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        return text.format(**kwargs) if kwargs else text

    def _handle_inappropriate(self, user_input: str) -> Optional[str]:
        text_lower = user_input.lower()
        if any(word in text_lower for word in INAPPROPRIATE_BLACKLIST):
            return self._get_localized_text("inappropriate_msg")
        return None

    def _handle_humor(self, user_input: str) -> Optional[str]:
        text_lower = user_input.lower()
        if any(ind in text_lower for ind in HUMOR_INDICATORS):
            return prompt.generate_humor_response(self.llm, self.user_language, user_input)
        return None

    def _generate_guiding_question(self, user_answer: str, question: str, context: str) -> str:
        return prompt.generate_guiding_question(
            llm=self.llm,
            user_language=self.user_language,
            chat_history=self.chat_history,
            question=question,
            answer=user_answer,
            context=context
        )

    def _generate_progressive_hint(self, hint_level: int = 0) -> Optional[str]:
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list)):
            hints = self.current_exercise["text"]["hint"]
            if hint_level < len(hints):
                hint_text = hints[hint_level].replace('$', '')
                return prompt.translate_text_to_english(self.llm, hint_text) if self.user_language == "en" else hint_text
        # Fallback to LLM-generated hint
        try:
            hint_prompt = prompt.get_guiding_question_prompt(self.user_language)
            hint_chain = hint_prompt | self.llm
            response = hint_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": self._get_current_question(),
                "answer": "",
                "context": "Generate a hint for the problem."
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return None

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str) -> Dict[str, Any]:
        evaluation_prompt = prompt.get_rag_prompt(self.user_language)
        evaluation_chain = evaluation_prompt | self.llm
        try:
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "context": context,
                "input": f"Evaluate if the answer '{user_input}' is correct for the question '{question}' with solution '{solution}'. Return 'CORRECT: [comment]' or 'INCORRECT: [explanation]'."
            })
            evaluation_result = eval_response.content.strip()
            is_correct = evaluation_result.lower().startswith("correct:")
            return {
                "is_correct": is_correct,
                "feedback": evaluation_result,
                "needs_guidance": not is_correct
            }
        except Exception as e:
            logger.error(f"Error in answer evaluation: {e}")
            return {
                "is_correct": False,
                "feedback": self._get_localized_text("wrong_answer"),
                "needs_guidance": True
            }

    def _handle_solution_request(self, user_input: str) -> str:
        lang_dict = I18N[self.user_language]
        if self.guidance_step < 4:  # Must go through reread, GQ1, GQ2, hint
            return lang_dict["need_more_attempts"]
        
        solution_prefix = lang_dict["solution_prefix"]
        solution = self._get_current_solution()
        current_question = self._get_current_question()
        
        explanation = prompt.generate_solution_explanation(
            llm=self.llm,
            user_language=self.user_language,
            chat_history=self.chat_history,
            question=current_question,
            solution=solution
        )
        
        svg_reference = ""
        if self.current_exercise and self.current_exercise.get("svg"):
            svg_reference = self._generate_and_save_svg(for_solution_explanation=True)
        
        self.exercise_count += 1
        result = f"{solution_prefix}{solution}\n\n{explanation}"
        if svg_reference:
            result += f"\n{svg_reference}"
        result += self._move_to_next_exercise_or_summary()
        return result

    def _handle_learning(self, user_input: str) -> str:
        lang_dict = I18N[self.user_language]
        
        # Detect intent using LLM
        current_question = self._get_current_question()
        intent = prompt.detect_intent(
            llm=self.llm,
            user_language=self.user_language,
            chat_history=self.chat_history,
            question=current_question,
            user_input=user_input
        )
        
        # Handle request for new exercise
        if intent == "new_exercise_request" or user_input.lower() in ["give me another exercise", "next exercise", "another one"]:
            self.exercise_count = min(self.exercise_count, MAX_EXERCISES - 1)  # Allow new exercise within limit
            query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
            if not self.current_exercise:
                return self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic) + "\n" + self._get_localized_text("no_relevant_exercises")
            self.state = State.LEARNING
            return f"{self._get_localized_text('continue_learning')}\n{self._get_current_question()}"
        
        if intent == "hint_request":
            if self.guidance_step < 3:  # Must go through reread, GQ1, GQ2
                return lang_dict["need_more_attempts"]
            self.guidance_step = 4
            self.state = State.PROVIDING_HINT
            hint = self._generate_progressive_hint(0)
            if hint:
                return f"{lang_dict['hint_prefix']}{hint}"
            self.guidance_step = 5
            return self._handle_solution_request(user_input)
        
        if intent == "solution_request":
            return self._handle_solution_request(user_input)
        
        # Default to answer_attempt
        current_solution = self._get_current_solution()
        retrieved_context = self.pinecone_index.query(
            vector=self.generate_embedding(f"Question: {current_question} User's Answer: {user_input}"),
            top_k=20,
            include_metadata=True,
            filter={"grade": {"$eq": self.hebrew_grade}, "topic": {"$eq": self.topic}} if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
        )
        context_str = "\n".join([c.metadata.get("text", "") for c in retrieved_context.matches if c.metadata.get("text")])
        if self.current_svg_description:
            context_str += f"\n\nImage Description: {self.current_svg_description}"
        
        evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, context_str)
        
        if evaluation_result["is_correct"]:
            self.exercise_count += 1
            self.guidance_step = 0
            self.state = State.LEARNING
            response = "✅ Correct!" + self._move_to_next_exercise_or_summary()
            self.chat_history.append(AIMessage(content=response))
            return response
        
        # Incorrect answer: progress through guidance steps
        self.guidance_step += 1
        feedback = evaluation_result["feedback"].split('\n')[0]
        
        if self.guidance_step == 1:
            self.state = State.REREAD_QUESTION
            response = f"{feedback}\n\n{lang_dict['reread_question']}"
        elif self.guidance_step == 2:
            self.state = State.GUIDING_QUESTION_1
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            response = f"{feedback}\n\n{lang_dict['encouragement']}{guiding_q}"
        elif self.guidance_step == 3:
            self.state = State.GUIDING_QUESTION_2
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            response = f"{feedback}\n\n{lang_dict['encouragement']}{guiding_q}"
        elif self.guidance_step == 4:
            self.state = State.PROVIDING_HINT
            hint = self._generate_progressive_hint(0)
            if hint:
                response = f"{feedback}\n\n{lang_dict['hint_prefix']}{hint}"
            else:
                self.guidance_step = 5
                response = self._handle_solution_request(user_input)
        else:  # guidance_step >= 5
            response = self._handle_solution_request(user_input)
        
        self.chat_history.append(AIMessage(content=response))
        return response

    def _handle_opening(self, user_input: str) -> str:
        self.opening_turn += 1
        lang_dict = I18N[self.user_language]
        
        # Validate input for meaningful content
        if self.opening_turn > 1 and (not user_input.strip() or len(user_input.strip()) < 3 or not any(c.isalpha() for c in user_input)):
            self.opening_turn -= 1
            try:
                retry_prompt = self.small_talk_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input,
                    "context": f"Generate a friendly prompt to ask the user to clarify their response for turn {self.opening_turn} in a small talk conversation."
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating retry prompt: {e}")
                retry_prompt = "What's up?" if self.user_language == "en" else "מה קורה?"
            response = self._get_localized_text("invalid_input", retry_prompt=retry_prompt)
            self.chat_history.append(AIMessage(content=response))
            return response

        if self.opening_turn == 1:
            response = "Hey hey, how are you?" if self.user_language == "en" else "היי היי, מה שלומך?"
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.opening_turn == 2:
            try:
                response = self.small_talk_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input,
                    "context": "Respond to the user's answer about how they are, then ask about their day in a friendly, engaging way."
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating small talk response: {e}")
                response = "Yes, I'm also good. Thanks for asking! How was your day? Long day?" if self.user_language == "en" else "כן, גם אני בסדר. תודה ששאלת! איך היה היום שלך? יום ארוך?"
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.opening_turn == 3:
            try:
                response = self.small_talk_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input,
                    "context": "Respond to the user's answer about their day, then ask about their hobbies in a friendly, engaging way."
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating small talk response: {e}")
                response = "Cool, sounds like a day! What hobbies do you have?" if self.user_language == "en" else "מגניב, נשמע כמו יום! אילו תחביבים יש לך?"
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.opening_turn == 4:
            response = self.personalized_followup_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "hobby": user_input
            }).content.strip()
            self.chat_history.append(AIMessage(content=response))
            return response
        else:  # Turn 5
            response = self.humorous_reaction_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input
            }).content.strip()
            self.chat_history.append(AIMessage(content=response))
            self.state = State.DIAGNOSTIC
            self.opening_turn = 0
            self.diagnostic_turn = 0
            return response

    def _generate_topic_suggestions(self, grade: str) -> str:
        try:
            topic_prompt = prompt.get_topic_suggestion_prompt(self.user_language)
            topic_chain = topic_prompt | self.llm
            response = topic_chain.invoke({
                "grade": grade,
                "context": f"Generate a list of 3-5 math topics suitable for grade {grade}."
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating topic suggestions: {e}")
            return "linear algebra, geometry, fractions"  # Fallback topics

    def _handle_diagnostic(self, user_input: str) -> str:
        self.diagnostic_turn += 1
        lang_dict = I18N[self.user_language]
        
        if self.diagnostic_turn == 1:
            response = lang_dict["ask_grade"]
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.diagnostic_turn == 2:
            if user_input not in SUPPORTED_GRADES:
                response = lang_dict["invalid_grade"]
                self.diagnostic_turn -= 1  # Retry grade selection
                self.chat_history.append(AIMessage(content=response))
                return response
            self.grade = user_input
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            topics = self._generate_topic_suggestions(self.grade)
            response = lang_dict["ask_topic"].format(grade=self.grade, topics=topics)
            self.chat_history.append(AIMessage(content=response))
            return response
        else:
            self.topic = user_input
            query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
            if not self.current_exercise:
                topics = self._generate_topic_suggestions(self.grade)
                response = f"{self._get_localized_text('no_exercises', grade=self.grade, topic=self.topic)}\n{self._get_localized_text('invalid_topic', topic=self.topic, topics=topics)}"
                self.diagnostic_turn -= 1  # Retry topic selection
                self.chat_history.append(AIMessage(content=response))
                return response
            self.state = State.LEARNING
            response = f"{self._get_localized_text('ready_for_question')}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response))
            return response

    def _handle_summary(self, user_input: str) -> str:
        lang_dict = I18N[self.user_language]
        # Check if user wants to continue with more exercises
        if user_input.lower() in ["not bye", "continue", "more", "give me another exercise", "next exercise", "another one"]:
            self.exercise_count = min(self.exercise_count, MAX_EXERCISES - 1)  # Allow new exercise within limit
            query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
            if not self.current_exercise:
                response = f"{self._get_localized_text('no_exercises', grade=self.grade, topic=self.topic)}\n{self._get_localized_text('invalid_topic', topic=self.topic, topics=self._generate_topic_suggestions(self.grade))}"
                self.chat_history.append(AIMessage(content=response))
                return response
            self.state = State.LEARNING
            response = f"{self._get_localized_text('continue_learning')}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response))
            return response
        
        summary = prompt.generate_lesson_summary(
            llm=self.llm,
            user_language=self.user_language,
            chat_history=self.chat_history,
            diagnostic={"grade": self.grade, "topic": self.topic}
        )
        closing = self._get_localized_text("closing_message")
        response = f"{summary}\n\n{closing}"
        self.chat_history.append(AIMessage(content=response))
        return response

    def _reset_guidance(self):
        self.guidance_step = 0
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        return next((ex for ex in self.exercises_data if ex.get("canonical_exercise_id") == exercise_id), None)

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        relevant_chunks = self.pinecone_index.query(
            vector=self.generate_embedding(query),
            top_k=20,
            include_metadata=True,
            filter={"grade": {"$eq": grade}, "topic": {"$eq": topic}} if grade and topic else None
        )
        if not relevant_chunks.matches:
            self.current_exercise = None
            self.current_svg_description = None
            return

        all_exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks.matches))
        available_exercise_ids = [ex_id for ex_id in all_exercise_ids if ex_id not in self.recently_asked_exercise_ids]
        if not available_exercise_ids:
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

        self.current_hint_index = 0
        self.current_question_index = 0
        self.current_svg_description = None
        self._reset_guidance()
        self.recently_asked_exercise_ids.append(chosen_exercise_id)
        if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
            self.recently_asked_exercise_ids.pop(0)

        if self.current_exercise.get("svg"):
            try:
                svg_content = self.current_exercise["svg"][0]
                self.current_svg_description = prompt.describe_svg_content(self.llm, svg_content)
            except Exception as e:
                logger.error(f"Error processing SVG for exercise {chosen_exercise_id}: {e}")
                self.current_svg_description = "Image description unavailable."

    def _generate_and_save_svg(self, for_solution_explanation: bool = False) -> str:
        if not (self.current_exercise and self.current_exercise.get("svg")):
            return ""
        try:
            svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
            svg_content = self.current_exercise["svg"][svg_content_idx]
            if not svg_content:
                return ""
            svg_filename = f"exercise_{self.current_exercise['canonical_exercise_id']}_q{self.current_question_index}{'solution' if for_solution_explanation else ''}_{uuid.uuid4().hex[:8]}.svg"
            svg_filepath = SVG_OUTPUT_DIR / svg_filename
            with open(svg_filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)
            if not for_solution_explanation:
                self.current_svg_file_path = svg_filepath
            file_reference = f"\n\n[Image File: {svg_filepath.as_posix()}]"
            if self.current_svg_description:
                file_reference += f"\n[Image Description: {self.current_svg_description}]"
            return file_reference
        except Exception as e:
            logger.error(f"Error processing SVG: {e}")
            return ""

    def _get_current_question(self) -> str:
        if not (self.current_exercise and 
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list)):
            logger.warning("Invalid exercise data structure.")
            return "No question available."
        
        questions = self.current_exercise["text"]["question"]
        if not (0 <= self.current_question_index < len(questions)):
            logger.warning(f"Invalid question index: {self.current_question_index}")
            return "No question available."
        
        q_text = questions[self.current_question_index].replace(',', '')
        if self.current_exercise.get("svg") and not self.svg_generated_for_question:
            svg_reference = self._generate_and_save_svg(for_solution_explanation=False)
            q_text += svg_reference
            self.svg_generated_for_question = True
        elif self.current_svg_file_path and self.svg_generated_for_question:
            q_text += f"\n\n[Image File: {self.current_svg_file_path.as_posix()}]"
            if self.current_svg_description:
                q_text += f"\n[Image Description: {self.current_svg_description}]"
        
        return prompt.translate_text_to_english(self.llm, q_text) if self.user_language == "en" else q_text

    def _get_current_solution(self) -> str:
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index].replace(',', '')
            return prompt.translate_text_to_english(self.llm, sol_text) if self.user_language == "en" else sol_text
        # Fallback to LLM-generated solution
        try:
            solution_prompt = prompt.get_solution_explanation_prompt(self.user_language)
            solution_chain = solution_prompt | self.llm
            response = solution_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": self._get_current_question(),
                "solution": ""
            })
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return "No solution available."

    def _move_to_next_exercise_or_summary(self) -> str:
        if self.exercise_count >= MAX_EXERCISES:
            self.state = State.SUMMARY
            return ""
        
        if (self.current_exercise and
            "text" in self.current_exercise and
            "question" in self.current_exercise["text"] and
            isinstance(self.current_exercise["text"]["question"], list) and
            self.current_question_index < len(self.current_exercise["text"]["question"]) - 1):
            self.current_question_index += 1
            self._reset_guidance()
            return f"\n\nNext question:\n{self._get_current_question()}"
        
        query = f"Next exercise for grade {self.hebrew_grade} on topic {self.topic}"
        self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
        if not self.current_exercise:
            self.state = State.SUMMARY
            return ""
        
        return f"\n\nNext exercise:\n{self._get_current_question()}"

    def transition(self, user_input: str) -> str:
        if user_input.strip():
            self.inactivity_timer.reset()
        
        text_lower = (user_input or "").strip().lower()
        if user_input:
            detected_lang = self.detect_language(user_input)
            if detected_lang != self.user_language and detected_lang in ["he", "en"]:
                self.user_language = detected_lang
                # Update prompt chains
                self.rag_chain = prompt.get_rag_prompt(self.user_language) | self.llm
                self.small_talk_chain = prompt.get_small_talk_prompt(self.user_language) | self.llm
                self.personal_followup_chain = prompt.get_personal_followup_prompt(self.user_language) | self.llm
                self.academic_transition_chain = prompt.get_academic_transition_prompt(self.user_language) | self.llm
                self.personalized_followup_chain = prompt.get_personalized_followup_prompt(self.user_language) | self.llm
                self.humorous_reaction_chain = prompt.get_humorous_reaction_prompt(self.user_language) | self.llm
        
        if user_input:
            self.chat_history.append(HumanMessage(content=user_input))
        
        inappropriate = self._handle_inappropriate(user_input)
        if inappropriate:
            self.chat_history.append(AIMessage(content=inappropriate))
            return inappropriate
        
        humor = self._handle_humor(user_input)
        if humor:
            self.chat_history.append(AIMessage(content=humor))
        
        if self.state == State.OPENING:
            response = self._handle_opening(user_input)
        elif self.state == State.DIAGNOSTIC:
            response = self._handle_diagnostic(user_input)
        elif self.state in [State.LEARNING, State.REREAD_QUESTION, State.GUIDING_QUESTION_1, State.GUIDING_QUESTION_2, State.PROVIDING_HINT]:
            response = self._handle_learning(user_input)
        elif self.state == State.SUMMARY:
            response = self._handle_summary(user_input)
            if user_input.lower() in {"exit", "quit", "done"}:
                return "👋 Bye!"
        else:
            response = "I'm not sure how to proceed. Type 'exit' to quit."
        
        self.chat_history.append(AIMessage(content=response))
        return response

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect if text is Hebrew or English."""
        return "he" if prompt.is_likely_hebrew(text) else "en"

    @staticmethod
    def generate_embedding(text: str, embedding_model) -> List[float]:
        """Generate embedding for a given text using SentenceTransformer."""
        if embedding_model is None:
            logger.error("Embedding model not loaded.")
            return []
        try:
            return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []