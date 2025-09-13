import os
import json
import random
import time
import threading
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import uuid
import prompt
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Inactivity and Interaction Settings
INACTIVITY_TIMEOUT = 60
TYPING_DETECTION_THRESHOLD = 5
INAPPROPRIATE_BLACKLIST = ["violence", "politics", "drugs", "sex", "hate", "terrorism"]
HUMOR_INDICATORS = ["haha", "lol", "😂", "🤣", "funny"]
MAX_EXERCISES = 2  # Exactly 2 exercises for PoC
SUPPORTED_GRADES = ["7", "8"]

# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7 or 8)",
        "invalid_grade": "Sorry, I only support grades 7 and 8 for now. Please choose one (e.g., 7 or 8).",
        "ask_topic": "Great! Grade {grade}. Which topic would you like to practice? Here are some suggestions: {topics}",
        "invalid_topic": "Hmm, that topic is too vague or not specific enough. Please choose one of these: {topics}",
        "invalid_input": "Hmm, I didn't quite catch that! {retry_prompt}",
        "ready_for_question": "Awesome! Let's start with this exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "reread_question": "Try reading the question again carefully…",
        "encouragement": "You're on the right track! ",
        "try_again": "Can you try again? Think about your approach.",
        "need_more_attempts": "Give it another try first - I believe you can work through this!",
        "ask_for_solution": "Would you like me to show you the solution?",
        "inappropriate_msg": "Sorry, but discussing these topics is not allowed during the lesson.",
        "inactivity_check": "Hey, are you still there?",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "closing_message": "Great, that was an awesome lesson! I'll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you're a genius. Bye!",
        "continue_learning": "Awesome, let's keep going! Here's another exercise:",
        "exercise_complete": "Great job! That's exercise {current} of {total}.",
        "next_exercise": "Ready for the next one? Here it is:"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז או ח)",
        "invalid_grade": "מצטער, אני תומך רק בכיתות ז ו-ח כרגע. אנא בחר אחת (למשל, ז או ח).",
        "ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? הנה כמה הצעות: {topics}",
        "invalid_topic": "המממ, הנושא הזה מעורפל מדי או לא ספציפי מספיק. אנא בחר אחד מאלה: {topics}",
        "invalid_input": "המממ, לא ממש הבנתי! {retry_prompt}",
        "ready_for_question": "מעולה! בואו נתחיל עם התרגיל הזה:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "reread_question": "נסה לקרוא את השאלה שוב בקפידה…",
        "encouragement": "אתה בכיוון הנכון! ",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
        "need_more_attempts": "תן לזה עוד ניסיון - אני מאמין שאתה יכול לעבוד על זה!",
        "ask_for_solution": "האם תרצה שאראה לך את הפתרון?",
        "inappropriate_msg": "מצטער, אבל דיון בנושאים אלה אסור במהלך השיעור.",
        "inactivity_check": "היי, אתה עדיין שם?",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "closing_message": "שיעור נהדר! אשלח לך תרגילים דומים לתרגול ונפגש בשיעור הבא. אם יש לך שאלות, תרגיש חופשי לשלוח לי הודעה. ואם אתה נתקע - זכור, אתה גאון. ביי!",
        "continue_learning": "מעולה, בוא נמשיך! הנה תרגיל נוסף:",
        "exercise_complete": "כל הכבוד! זה תרגיל {current} מתוך {total}.",
        "next_exercise": "מוכן לעוד אחד? הנה הוא:"
    }
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    OPENING = auto()
    DIAGNOSTIC = auto()
    LEARNING = auto()
    SUMMARY = auto()
    REREAD_QUESTION = auto()
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
    def __init__(self, exercises_data, pinecone_index, llm, embedding_model):
        self.state = State.OPENING
        self.grade = None
        self.hebrew_grade = None
        self.topic = None
        self.exercises_data = exercises_data
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model
        self.chat_history = []
        self.current_svg_description = None
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 0
        self.exercise_count = 0
        self.guidance_step = 0
        self.opening_turn = 0
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

    def _get_exercise_context(self, question: str, user_input: str) -> str:
        """Retrieve relevant context for the current exercise"""
        try:
            retrieved_context = self.pinecone_index.query(
                vector=self.generate_embedding(f"Question: {question} User's Answer: {user_input}"),
                top_k=20,
                include_metadata=True,
                filter={"grade": {"$eq": self.hebrew_grade}, "topic": {"$eq": self.topic}} if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            context_str = "\n".join([c.metadata.get("text", "") for c in retrieved_context.matches if c.metadata.get("text")])
            if self.current_svg_description:
                context_str += f"\n\nImage Description: {self.current_svg_description}"
            return context_str
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def _generate_guiding_question(self, user_answer: str, question: str, context: str) -> str:
        """Generate contextual guiding questions with LLM fallback"""
        try:
            guiding_prompt = prompt.get_guiding_question_prompt(self.user_language)
            guiding_chain = guiding_prompt | self.llm
            
            response = guiding_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "user_answer": user_answer,
                "context": context,
                "guidance_step": self.guidance_step,
                "topic": self.topic
            })
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            
            # Fallback questions based on guidance step
            if self.guidance_step == 2:
                return "What information do you have, and what are you trying to find?" if self.user_language == "en" else "איזה מידע יש לך, ומה אתה מחפש למצוא?"
            else:  # guidance_step == 3
                return "Can you identify the key steps needed to solve this?" if self.user_language == "en" else "תוכל לזהות את השלבים המרכזיים הנדרשים לפתרון?"

    def _get_or_generate_hint(self) -> str:
        """Get hint from dataset or generate via LLM if missing"""
        
        # Try to get hint from dataset first
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list) and
            self.current_hint_index < len(self.current_exercise["text"]["hint"])):
            
            hint_text = self.current_exercise["text"]["hint"][self.current_hint_index]
            self.current_hint_index += 1
            return prompt.translate_text_to_english(self.llm, hint_text.replace('$', '')) if self.user_language == "en" else hint_text.replace('$', '')
        
        # Generate hint via LLM if missing from dataset
        try:
            hint_prompt = prompt.get_hint_generation_prompt(self.user_language)
            hint_chain = hint_prompt | self.llm
            response = hint_chain.invoke({
                "question": self._get_current_question(),
                "solution": self._get_current_solution(),
                "chat_history": self.chat_history[-3:],
                "hint_level": self.current_hint_index,
                "topic": self.topic
            })
            self.current_hint_index += 1
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "Think about breaking this problem into smaller steps." if self.user_language == "en" else "נסה לפרק את הבעיה לשלבים קטנים יותר."

    def _generate_new_exercise(self, grade: str, topic: str) -> Dict[str, Any]:
        try:
            exercise_prompt = prompt.get_exercise_prompt(self.user_language)
            exercise_chain = exercise_prompt | self.llm
            response = exercise_chain.invoke({
                "chat_history": self.chat_history[-5:],
                "grade": grade,
                "topic": topic
            })
            exercise_data = json.loads(response.content.strip())
            return exercise_data
        except Exception as e:
            logger.error(f"Error generating new exercise: {e}")
            return {
                "text": {
                    "question": [f"Calculate the result of a simple {topic} problem for grade {grade}, e.g., add two rational numbers for grade {grade}."],
                    "solution": [f"Sample solution for a {topic} problem."],
                    "hint": [f"Think about the key steps in {topic} for grade {grade}."]
                },
                "grade": grade,
                "topic": topic
            }

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str) -> Dict[str, Any]:
        """Evaluate user answer and provide guidance feedback"""
        try:
            evaluation_prompt = prompt.get_answer_evaluation_prompt(self.user_language)
            evaluation_chain = evaluation_prompt | self.llm
            
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "context": context,
                "question": question,
                "user_answer": user_input,
                "correct_solution": solution,
                "topic": self.topic
            })
            
            evaluation_result = eval_response.content.strip()
            is_correct = "correct" in evaluation_result.lower()
            
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

    def _handle_correct_answer(self) -> str:
        """Handle correct answer with exercise progression"""
        lang_dict = I18N[self.user_language]
        
        self.exercise_count += 1
        self._reset_guidance()
        
        response = f"✅ Correct!\n\n{lang_dict['exercise_complete'].format(current=self.exercise_count, total=MAX_EXERCISES)}"
        
        if self.exercise_count >= MAX_EXERCISES:
            # Transition to summary after exactly 2 exercises
            self.state = State.SUMMARY
            response += f"\n\n{self._generate_lesson_summary()}"
        else:
            # Move to next exercise
            response += self._move_to_next_exercise_or_summary()
        
        return response

    def _handle_incorrect_answer(self, evaluation_result, user_input, current_question, context_str):
        """Handle incorrect answer with 5-step misunderstanding mechanism"""
        lang_dict = I18N[self.user_language]
        
        self.guidance_step += 1
        feedback = evaluation_result["feedback"].split('\n')[0]
        
        if self.guidance_step == 1:
            # Step 1: "Try reading the question again carefully..."
            self.state = State.REREAD_QUESTION
            response = f"{feedback}\n\n{lang_dict['reread_question']}"
            
        elif self.guidance_step == 2:
            # Step 2: First Guiding Question
            self.state = State.GUIDING_QUESTION_1
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            response = f"{feedback}\n\n{lang_dict['encouragement']}{guiding_q}"
            
        elif self.guidance_step == 3:
            # Step 3: Second Guiding Question
            self.state = State.GUIDING_QUESTION_2
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            response = f"{feedback}\n\n{lang_dict['encouragement']}{guiding_q}"
            
        elif self.guidance_step == 4:
            # Step 4: Provide Hint
            self.state = State.PROVIDING_HINT
            hint = self._get_or_generate_hint()
            response = f"{feedback}\n\n{lang_dict['hint_prefix']}{hint}"
            
        else:  # guidance_step >= 5
            # Step 5: Full solution and move to next exercise
            return self._provide_full_solution_and_progress()
        
        return response

    def _provide_full_solution_and_progress(self) -> str:
        """Provide full solution and automatically progress to next exercise"""
        lang_dict = I18N[self.user_language]
        
        solution_prefix = lang_dict["solution_prefix"]
        solution = self._get_current_solution()
        
        # Generate explanation via LLM if needed
        explanation = self._get_or_generate_solution_explanation()
        
        # Include SVG reference if available
        svg_reference = ""
        if self.current_exercise and self.current_exercise.get("svg"):
            svg_reference = self._generate_and_save_svg(for_solution_explanation=True)
        
        self.exercise_count += 1
        self._reset_guidance()
        
        response = f"{solution_prefix}{solution}\n\n{explanation}"
        if svg_reference:
            response += f"\n{svg_reference}"
        
        if self.exercise_count >= MAX_EXERCISES:
            # Transition to summary after exactly 2 exercises
            self.state = State.SUMMARY
            response += f"\n\n{self._generate_lesson_summary()}"
        else:
            # Move to next exercise
            response += self._move_to_next_exercise_or_summary()
        
        return response

    def _get_or_generate_solution_explanation(self) -> str:
        """Get solution explanation from dataset or generate via LLM if missing"""
        try:
            explanation_prompt = prompt.get_solution_explanation_prompt(self.user_language)
            explanation_chain = explanation_prompt | self.llm
            
            response = explanation_chain.invoke({
                "question": self._get_current_question(),
                "solution": self._get_current_solution(),
                "topic": self.topic,
                "grade": self.grade,
                "chat_history": self.chat_history[-3:]
            })
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating solution explanation: {e}")
            return "Here's the step-by-step approach to solve this problem." if self.user_language == "en" else "הנה הגישה השלב אחר שלב לפתרון הבעיה."

    def _handle_solution_request(self, user_input: str) -> str:
        """Handle explicit solution requests"""
        lang_dict = I18N[self.user_language]
        if self.guidance_step < 4:
            return lang_dict["need_more_attempts"]
        
        return self._provide_full_solution_and_progress()

    def _provide_hint(self) -> str:
        """Provide hint if requirements are met"""
        lang_dict = I18N[self.user_language]
        self.guidance_step = 4
        self.state = State.PROVIDING_HINT
        hint = self._get_or_generate_hint()
        return f"{lang_dict['hint_prefix']}{hint}"

    def _handle_exercise_progression(self) -> str:
        """Handle requests for new exercises within the 2-exercise limit"""
        if self.exercise_count >= MAX_EXERCISES:
            self.state = State.SUMMARY
            return self._generate_lesson_summary()
        
        # Pick new exercise via RAG or generate
        query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
        self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
        
        if not self.current_exercise:
            self.current_exercise = self._generate_new_exercise(self.hebrew_grade, self.topic)
        
        self.state = State.LEARNING
        return f"{self._get_localized_text('continue_learning')}\n{self._get_current_question()}"

    def _handle_learning(self, user_input: str) -> str:
        """Main learning state handler with refined 5-step mechanism"""
        lang_dict = I18N[self.user_language]
        
        current_question = self._get_current_question()
        intent = prompt.detect_intent(
            llm=self.llm,
            user_language=self.user_language,
            chat_history=self.chat_history,
            question=current_question,
            user_input=user_input
        )
        
        # Handle explicit requests first
        if intent == "new_exercise_request" or user_input.lower() in ["give me another exercise", "next exercise", "another one"]:
            return self._handle_exercise_progression()
        
        if intent == "hint_request":
            if self.guidance_step < 3:  # Must attempt 3 times before hint
                return lang_dict["need_more_attempts"]
            return self._provide_hint()
        
        if intent == "solution_request":
            return self._handle_solution_request(user_input)
        
        # Main answer evaluation
        current_solution = self._get_current_solution()
        retrieved_context = self._get_exercise_context(current_question, user_input)
        
        evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, retrieved_context)
        
        if evaluation_result["is_correct"]:
            response = self._handle_correct_answer()
            self.chat_history.append(AIMessage(content=response))
            return response
        else:
            response = self._handle_incorrect_answer(evaluation_result, user_input, current_question, retrieved_context)
            self.chat_history.append(AIMessage(content=response))
            return response

    def _handle_opening(self, user_input: str) -> str:
        self.opening_turn += 1
        lang_dict = I18N[self.user_language]
        
        vague_responses = ["yes", "no", "not curious", "nothing", "idk", "dunno", "none", "n/a", "whatever"]
        
        if self.opening_turn > 1 and (
            not user_input.strip() or
            len(user_input.strip()) < 3 or
            not any(c.isalpha() for c in user_input) or
            user_input.lower() in vague_responses
        ):
            self.opening_turn -= 1
            try:
                retry_context = {
                    2: "The user responded to 'how are you?' with a vague or non-informative answer. Ask them to clarify how they are feeling in a friendly, engaging way, ensuring the response stays in the small talk phase and does not mention math or academic topics.",
                    3: "The user responded to 'how was your day?' with a vague or non-informative answer. Ask them to clarify about their day in a friendly, engaging way, ensuring the response stays in the small talk phase and does not mention math or academic topics.",
                    4: "The user responded to 'what hobbies do you have?' with a vague or non-informative answer. Ask them to clarify their hobbies in a friendly, engaging way, ensuring the response stays in the small talk phase and does not mention math or academic topics."
                }.get(self.opening_turn, "Ask the user to clarify their response in a friendly, engaging way, ensuring the response stays in the small talk phase and does not mention math or academic topics.")
                retry_prompt = self.small_talk_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input,
                    "context": retry_context
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating retry prompt: {e}")
                retry_prompt = {
                    2: "Could you share a bit more about how you're feeling today? Excited, relaxed, or something else?",
                    3: "Tell me more about your day! Was it busy, fun, or maybe super chill?",
                    4: "What's a hobby you're into or something fun you did recently?"
                }.get(self.opening_turn, "Could you share a bit more clearly?")
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
                    "context": "Acknowledge the user's response about how they are, then ask about their day in a friendly, engaging way. Do not mention math or academic topics."
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating small talk response: {e}")
                response = "Cool, glad you're doing alright! How was your day? Long day?" if self.user_language == "en" else "מגניב, שמח שאתה בסדר! איך היה היום שלך? יום ארוך?"
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.opening_turn == 3:
            try:
                response = self.small_talk_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input,
                    "context": "Acknowledge the user's response about their day, then ask about their hobbies in a friendly, engaging way. Do not mention math or academic topics."
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating small talk response: {e}")
                response = "Nice, sounds like a day! What hobbies do you have?" if self.user_language == "en" else "מגניב, נשמע כמו יום! אילו תחביבים יש לך?"
            self.chat_history.append(AIMessage(content=response))
            return response
        elif self.opening_turn == 4:
            try:
                response = self.personalized_followup_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "hobby": user_input
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating personalized follow-up: {e}")
                response = f"Wow, {user_input} sounds cool! Tell me more about it!" if self.user_language == "en" else f"וואו, {user_input} נשמע מגניב! ספר לי עוד על זה!"
            self.chat_history.append(AIMessage(content=response))
            return response
        else:  # Turn 5
            try:
                response = self.humorous_reaction_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "input": user_input
                }).content.strip()
            except Exception as e:
                logger.error(f"Error generating humorous reaction: {e}")
                response = "Haha, you're full of surprises! Let's get those brain gears turning!" if self.user_language == "en" else "חהחה, אתה מלא בהפתעות! בוא נתחיל להפעיל את גלגלי המוח!"
            self.chat_history.append(AIMessage(content=response))
            self.state = State.DIAGNOSTIC
            self.opening_turn = 0
            self.diagnostic_turn = 1  # Start with asking for grade
            return response + "\n" + self._get_localized_text("ask_grade")

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
            if grade == "7":
                return """1. **Ratios, Proportions, and Percentages**: Understanding unit rates, solving proportional relationships, calculating percentages (including discounts, taxes, and interest), and working with scale drawings.
2. **Operations with Rational Numbers**: Adding, subtracting, multiplying, and dividing positive and negative integers, fractions, and decimals. This also includes understanding absolute value.
3. **Solving Linear Equations and Inequalities**: Setting up and solving one- and two-step equations and inequalities with rational coefficients, and representing solutions on a number line.
4. **Geometry: Area, Volume, and Angle Relationships**: Calculating the area and circumference of circles, the area of composite figures, and the volume and surface area of 3D shapes (prisms, pyramids). Understanding angle relationships (complementary, supplementary, vertical, adjacent)."""
            else:  # grade 8
                return """1. **Solving Multi-Step Linear Equations and Inequalities**: Equations with variables on both sides, distributive property, and solving and graphing simple inequalities.
2. **The Pythagorean Theorem**: Applying the theorem to find missing side lengths in right triangles and determine distances on a coordinate plane.
3. **Introduction to Functions**: Identifying functions, understanding input/output, representing functions with tables, graphs, and equations, and exploring linear functions.
4. **Geometric Transformations**: Exploring translations, reflections, rotations, and dilations on the coordinate plane, and understanding congruence and similarity.
5. **Finding Slope and Y-Intercept**: Determining the slope and y-intercept of a line from its equation or graph."""
    
    def _handle_diagnostic(self, user_input: str) -> str:
        lang_dict = I18N[self.user_language]
        
        logger.debug(f"Handling diagnostic: turn={self.diagnostic_turn}, input='{user_input}'")
        
        if self.diagnostic_turn == 1:
            normalized_input = str(user_input).strip()
            logger.debug(f"Normalized input: '{normalized_input}', SUPPORTED_GRADES={SUPPORTED_GRADES}")
            if normalized_input not in SUPPORTED_GRADES:
                response = lang_dict["invalid_grade"]
                self.chat_history.append(AIMessage(content=response))
                logger.debug(f"Invalid grade input: {response}")
                return response
            self.grade = normalized_input
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.diagnostic_turn = 2
            topics = self._generate_topic_suggestions(self.grade)
            response = lang_dict["ask_topic"].format(grade=self.grade, topics=topics)
            self.chat_history.append(AIMessage(content=response))
            logger.debug(f"Valid grade {self.grade}, prompting for topic: {response}")
            return response
        elif self.diagnostic_turn == 2:
            self.topic = user_input.strip()
            topics_list = self._generate_topic_suggestions(self.grade).split('\n')
            valid_topics = [t.lower().split(':')[0].split('**')[-1].strip() for t in topics_list if t.strip()]
            if self.topic.lower() in ["anyone", "any", "anything", "random", "whatever", "any topic"]:
                self.topic = random.choice(valid_topics) if valid_topics else "Operations with Rational Numbers" if self.grade == "7" else "Finding Slope and Y-Intercept"
                logger.debug(f"Vague topic '{user_input}' replaced with: {self.topic}")
            elif self.topic.lower() not in valid_topics:
                response = lang_dict["invalid_topic"].format(topic=user_input, topics=self._generate_topic_suggestions(self.grade))
                self.chat_history.append(AIMessage(content=response))
                logger.debug(f"Invalid topic input: {response}")
                return response
            query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
            if not self.current_exercise:
                self.current_exercise = self._generate_new_exercise(self.hebrew_grade, self.topic)
            self.state = State.LEARNING
            self.diagnostic_turn = 0
            response = f"{self._get_localized_text('ready_for_question')}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response))
            logger.debug(f"Topic selected: {self.topic}, transitioning to LEARNING: {response}")
            return response
        else:
            logger.error(f"Unexpected diagnostic_turn: {self.diagnostic_turn}")
            self.diagnostic_turn = 1
            response = lang_dict["invalid_input"].format(retry_prompt="Please enter your grade (e.g., 7 or 8).")
            self.chat_history.append(AIMessage(content=response))
            return response

    def _generate_lesson_summary(self) -> str:
        """Generate concise lesson summary (2-3 sentences) + closing message"""
        try:
            summary_prompt = prompt.get_lesson_summary_prompt(self.user_language)
            summary_chain = summary_prompt | self.llm
            
            summary_response = summary_chain.invoke({
                "topic": self.topic,
                "grade": self.grade,
                "exercise_count": self.exercise_count,
                "chat_history": self.chat_history[-10:]  # Include more context for summary
            })
            
            summary = summary_response.content.strip()
            closing = self._get_localized_text("closing_message")
            
            return f"{summary}\n\n{closing}"
            
        except Exception as e:
            logger.error(f"Error generating lesson summary: {e}")
            
            # Fallback summary
            lang_dict = I18N[self.user_language]
            fallback_summary = f"We completed {self.exercise_count} exercises on {self.topic} for grade {self.grade}. You worked through the key concepts step by step." if self.user_language == "en" else f"השלמנו {self.exercise_count} תרגילים ב{self.topic} לכיתה {self.grade}. עבדת על המושגים המרכזיים צעד אחר צעד."
            return f"{fallback_summary}\n\n{lang_dict['closing_message']}"

    def _handle_summary(self, user_input: str) -> str:
        lang_dict = I18N[self.user_language]
        if user_input.lower() in ["not bye", "continue", "more", "give me another exercise", "next exercise", "another one"]:
            if self.exercise_count < MAX_EXERCISES:
                query = f"Exercise for grade {self.hebrew_grade} on topic {self.topic}"
                self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
                if not self.current_exercise:
                    self.current_exercise = self._generate_new_exercise(self.hebrew_grade, self.topic)
                self.state = State.LEARNING
                response = f"{self._get_localized_text('continue_learning')}\n{self._get_current_question()}"
                self.chat_history.append(AIMessage(content=response))
                return response
        
        summary = self._generate_lesson_summary()
        response = summary
        self.chat_history.append(AIMessage(content=response))
        return response

    def _reset_guidance(self):
        self.guidance_step = 0
        self.current_hint_index = 0
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        return next((ex for ex in self.exercises_data if ex.get("canonical_exercise_id") == exercise_id), None)

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        try:
            relevant_chunks = self.pinecone_index.query(
                vector=self.generate_embedding(query),
                top_k=20,
                include_metadata=True,
                filter={"grade": {"$eq": grade}, "topic": {"$eq": topic}} if grade and topic else None
            )
            if not relevant_chunks.matches:
                self.current_exercise = self._generate_new_exercise(grade, topic)
                self.current_svg_description = None
                return

            all_exercise_ids = list(set(chunk.metadata.get("exercise_id") for chunk in relevant_chunks.matches if chunk.metadata.get("exercise_id")))
            available_exercise_ids = [ex_id for ex_id in all_exercise_ids if ex_id not in self.recently_asked_exercise_ids]
            if not available_exercise_ids:
                self.recently_asked_exercise_ids.clear()
                available_exercise_ids = all_exercise_ids
            if not available_exercise_ids:
                self.current_exercise = self._generate_new_exercise(grade, topic)
                self.current_svg_description = None
                return

            chosen_exercise_id = random.choice(available_exercise_ids)
            self.current_exercise = self._get_exercise_by_id(chosen_exercise_id)
            if not self.current_exercise:
                self.current_exercise = self._generate_new_exercise(grade, topic)
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
        except Exception as e:
            logger.error(f"Error in _pick_new_exercise_rag: {e}")
            self.current_exercise = self._generate_new_exercise(grade, topic)
            self.current_svg_description = None

    def _generate_and_save_svg(self, for_solution_explanation: bool = False) -> str:
        if not (self.current_exercise and self.current_exercise.get("svg")):
            return ""
        try:
            svg_content_idx = min(self.current_question_index, len(self.current_exercise["svg"]) - 1)
            svg_content = self.current_exercise["svg"][svg_content_idx]
            if not svg_content:
                return ""
            svg_filename = f"exercise_{self.current_exercise.get('canonical_exercise_id', 'unknown')}_q{self.current_question_index}{'_solution' if for_solution_explanation else ''}_{uuid.uuid4().hex[:8]}.svg"
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
        try:
            solution_prompt = prompt.get_solution_explanation_prompt(self.user_language)
            solution_chain = solution_prompt | self.llm
            response = solution_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": self._get_current_question(),
                "solution": "",
                "topic": self.topic
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
            return f"\n\n{self._get_localized_text('next_exercise')}\n{self._get_current_question()}"
        
        query = f"Next exercise for grade {self.hebrew_grade} on topic {self.topic}"
        self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=self.topic)
        if not self.current_exercise:
            self.current_exercise = self._generate_new_exercise(self.hebrew_grade, self.topic)
        
        return f"\n\n{self._get_localized_text('next_exercise')}\n{self._get_current_question()}"

    def transition(self, user_input: str) -> str:
        if user_input.strip():
            self.inactivity_timer.reset()
        
        text_lower = (user_input or "").strip().lower()
        if user_input:
            detected_lang = self.detect_language(user_input)
            if detected_lang != self.user_language and detected_lang in ["he", "en"]:
                self.user_language = detected_lang
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
            if user_input.lower() in ["exit", "quit", "done"]:
                return "👋 Bye!"
        else:
            response = "I'm not sure how to proceed. Type 'exit' to quit."
        
        if not humor:  # Only append if we didn't already append from humor handling
            self.chat_history.append(AIMessage(content=response))
        return response

    @staticmethod
    def detect_language(text: str) -> str:
        return "he" if prompt.is_likely_hebrew(text) else "en"

    def generate_embedding(self, text: str) -> List[float]:
        if self.embedding_model is None:
            logger.error("Embedding model not loaded.")
            return []
        try:
            return self.embedding_model.encode([text], show_progress_bar=False)[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []