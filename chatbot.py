# dialogue_multilang_rag_chat.py
import os
import json
import random
from pathlib import Path
from enum import Enum, auto
from google import genai

# -----------------------------
# CONFIG
# -----------------------------
PARSED_INPUT_FILE = Path("parsed_outputs/all_parsed.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# GenAI Chat Client
# -----------------------------
client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")

# -----------------------------
# Localization (English + Hebrew)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)\n2) Hebrew",
        "small_talk_prompts": [
            "Hi! How’s your day going?",
            "Did you watch the game yesterday?",
            "Hey! How’s everything today?"
        ],
        "personal_followup_prompts": [
            "How was your last class?",
            "Which topic did you enjoy the most recently?",
            "Was your last lesson easy or challenging?"
        ],
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
        "ask_topic": "Great! Grade {grade}. Which topic would you like to practice?",
        "ready_for_question": "Awesome! Let’s start with the next exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "❌ Incorrect. Try again or type 'hint' for a hint.",
        "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
        "pass_to_solution": "Moving to solution:"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית\n2) עברית (ברירת מחדל)",
        "small_talk_prompts": [
            "שלום! איך עובר עליך היום?",
            "צפית במשחק אתמול?",
            "היי! איך הכל היום?"
        ],
        "personal_followup_prompts": [
            "איך היה השיעור האחרון שלך?",
            "איזה נושא הכי נהנת לאחרונה?",
            "האם השיעור האחרון היה קל או מאתגר?"
        ],
        "ask_grade": "נחמד! לפני שנתחיל, באיזה כיתה אתה? (לדוגמה, 7, 8)",
        "ask_topic": "מעולה! כיתה {grade}. איזה נושא תרצה לתרגל?",
        "ready_for_question": "נהדר! בוא נתחיל עם התרגיל הבא:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "❌ לא נכון. נסה שוב או הקלד 'hint' לקבלת רמז.",
        "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
        "pass_to_solution": "מעבר לפתרון:"
    }
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
    END = auto()

# -----------------------------
# Helpers
# -----------------------------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def genai_chat_answer(message: str) -> str:
    """Send a message to Gemini chat and get answer."""
    try:
        response = chat.send_message(message)
        return response.text
    except Exception as e:
        return f"❌ GenAI Error: {str(e)}"

# -----------------------------
# Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data):
        self.state = State.START
        self.grade = None
        self.topic = None
        self.language = "en"  # default
        self.exercises_data = exercises_data
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0

    def _pick_new_exercise(self):
        if not self.exercises_data:
            self.current_exercise = None
            return
        self.current_exercise = random.choice(self.exercises_data)
        self.current_hint_index = 0

    def _get_current_question(self):
        q = self.current_exercise["text"]["question"][self.current_question_index]
        return q.replace("$", "")

    def _get_current_solution(self):
        s = self.current_exercise["text"]["solution"][self.current_question_index]
        return s.replace("$", "")

    def _get_current_hint(self):
        hints = self.current_exercise.get("hints", [])
        if hints:
            hint = hints[min(self.current_hint_index, len(hints)-1)]
            self.current_hint_index += 1
            return hint
        return None

    def transition(self, user_input: str) -> str:
        text = (user_input or "").strip()

        if self.state == State.START:
            self.state = State.SMALL_TALK
            return I18N["en"]["choose_language"]

        elif self.state == State.SMALL_TALK:
            # Set language based on user input
            if text == "2" or "hebrew" in text.lower() or "עברית" in text:
                self.language = "he"
            else:
                self.language = "en"
            self.state = State.PERSONAL_FOLLOWUP
            return random.choice(I18N[self.language]["small_talk_prompts"])

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.ASK_GRADE
            return random.choice(I18N[self.language]["personal_followup_prompts"]) + "\n" + I18N[self.language]["ask_grade"]

        elif self.state == State.ASK_GRADE:
            self.grade = text
            self.state = State.EXERCISE_SELECTION
            return I18N[self.language]["ask_topic"].format(grade=self.grade)

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = text.lower()
            self.state = State.QUESTION_ANSWER
            self._pick_new_exercise()
            if not self.current_exercise:
                return I18N[self.language]["no_exercises"].format(grade=self.grade, topic=self.topic)
            return f"{I18N[self.language]['ready_for_question']}\n{self._get_current_question()}"

        elif self.state == State.QUESTION_ANSWER:
            if text.lower() == "hint":
                hint = self._get_current_hint()
                if hint:
                    return f"{I18N[self.language]['hint_prefix']} {hint}"
                else:
                    return f"{I18N[self.language]['hint_prefix']} No more hints available."

            elif text.lower() in {"solution", "pass"}:
                solution = self._get_current_solution()
                self._pick_new_exercise()
                if self.current_exercise:
                    return f"{I18N[self.language]['solution_prefix']} {solution}\n\nNext question:\n{self._get_current_question()}"
                else:
                    genai_resp = genai_chat_answer(self.topic)
                    return f"{I18N[self.language]['solution_prefix']} {solution}\n\nGenAI says:\n{genai_resp}\nNo more exercises."

            else:
                correct_answer = self._get_current_solution().strip().lower()
                if text.strip().lower() == correct_answer:
                    self._pick_new_exercise()
                    if self.current_exercise:
                        return f"✅ Correct!\n\nNext question:\n{self._get_current_question()}"
                    else:
                        genai_resp = genai_chat_answer(self.topic)
                        return f"✅ Correct!\nGenAI suggests:\n{genai_resp}\nNo more exercises."
                else:
                    genai_resp = genai_chat_answer(text)
                    return f"{I18N[self.language]['wrong_answer']}\nGenAI suggests:\n{genai_resp}"

        return "?"

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not PARSED_INPUT_FILE.exists():
        print("❌ Missing JSON file.")
        return

    exercises = load_json(PARSED_INPUT_FILE)
    fsm = DialogueFSM(exercises)
    print(fsm.transition(""))

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            break
        A_GUY = fsm.transition(user_input)
        print(f"A_GUY: {A_GUY}")

if __name__ == "__main__":
    main()
