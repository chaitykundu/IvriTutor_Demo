# prompt.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional
import json

# -----------------------------
# Prompt Templates
# -----------------------------

def get_rag_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the RAG chatbot prompt template for math tutoring.
    Args:
        user_language: 'en' or 'he' to set response language.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful Math AI tutor.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        - If Hebrew, use Right-to-Left (RTL) for conversational text, keep mathematical expressions Left-to-Right (LTR)
        
        Teaching Guidelines:
        - Never give direct answers immediately
        - Use a gradual assistance approach: encouragement → guiding questions → hints → solution
        - Ask guiding questions to help students think through problems
        - Build understanding step by step
        - Use provided context for accurate information
        - If context lacks crucial information, state what's missing
        - When providing hints, use EXACT TEXT from context when available
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context: {context}\n\nQuestion: {input}")
    ])

def get_small_talk_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the small talk prompt for initiating conversation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a friendly Math AI tutor starting a conversation.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        - Default to English if unclear
        
        Personality:
        - Warm, encouraging, approachable
        - Enthusiastic about math
        - Keep responses short (1-2 sentences)
        - Examples: "Hey! How are you doing today?", "Hi there! Good to see you?"
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

def get_personal_followup_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for personalized follow-up in small talk.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are continuing a casual conversation with a student.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Acknowledge their response warmly
        - Keep it brief and natural (1-2 sentences)
        - Show genuine interest
        - Gradually transition toward academic readiness
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

def get_academic_transition_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for transitioning from small talk to academic topics.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are transitioning from personal chat to academic topics.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        - For Hebrew: Use RTL for general text, keep math expressions LTR
        
        Guidelines:
        - Ask about recent learning or upcoming academic events
        - Examples: "What did you learn recently?", "When is your next exam?"
        - Keep it friendly, short (1 sentence), natural
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

def get_personalized_followup_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for hobby-based personalized follow-up in Opening state.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a Math AI tutor generating a personalized follow-up based on the student's hobby.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Create a short, friendly follow-up question based on the hobby
        - Example: If hobby is basketball, ask "Did you play basketball today?"
        - Keep it 1-2 sentences, conversational
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Student's hobby: {hobby}"),
    ])

def get_humorous_reaction_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for a humorous reaction in Opening state.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a Math AI tutor generating a short humorous reaction.
        
        Language Rules:
        - Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Reference the student's hobby or recent input
        - Keep it light and encouraging
        - Example: "Great! Let’s see if your brain is as fit as your legs."
        - 1-2 sentences max
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

def get_guiding_question_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for generating guiding questions during learning.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a Math AI tutor generating a guiding question.
        
        Language: Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Ask a question to guide the student toward the solution
        - Don't give away the answer
        - Focus on the mathematical concept or method
        - Be encouraging, supportive, concise (1-2 sentences)
        - For Hebrew, use RTL for conversational text, LTR for math
        - Examples: "What operation should we use first?", "What type of equation is this?"
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Problem: {question}\nStudent's Answer: {answer}\nContext: {context}\n\nGenerate a helpful guiding question:")
    ])

def get_humor_response_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for responding to detected humor (e.g., 'haha', 'lol').
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""Generate a short, friendly humorous response or emoji to match the user's laughter.
        
        Language: Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Keep it brief (1 sentence or emoji)
        - Match the user's tone (e.g., 'Haha, glad you're having fun! 😂')
        """),
        ("user", "{input}"),
    ])

def get_solution_explanation_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for generating a step-by-step solution explanation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a Math AI tutor providing a detailed solution explanation.
        
        Language: Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Explain the solution step by step
        - Show reasoning for each step
        - Help the student understand the concept
        - Be clear, educational, concise but thorough
        - Reference the image/graph when relevant
        - For Hebrew, use RTL for conversational text, LTR for math
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Question: {question}\nSolution: {solution}\n\nProvide a step-by-step explanation:")
    ])

def get_summary_prompt(user_language: str = "en") -> ChatPromptTemplate:
    """
    Returns the prompt for generating a lesson summary.
    """
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a Math AI tutor generating a concise lesson summary.
        
        Language: Respond in {'English' if user_language == 'en' else 'Hebrew'}
        
        Guidelines:
        - Summarize the lesson in 2-3 sentences
        - Personalize based on diagnostic answers and exercises
        - Be positive and encouraging
        - For Hebrew, use RTL for conversational text, LTR for math
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Diagnostic: {diagnostic}\nGenerate summary:")
    ])

def get_translation_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for translating text to English.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a precise translator. Translate the following text to English. If it's already in English, return it as is. Provide ONLY the translation."),
        ("user", "{input}"),
    ])

def get_svg_description_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for describing SVG content.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code. Provide only a brief description."),
        ("user", "Describe the following SVG content:\n```svg\n{svg_input}\n```"),
    ])

# -----------------------------
# Prompt-Related Functions
# -----------------------------

def generate_guiding_question(
    llm,
    user_language: str,
    chat_history: List,
    question: str,
    answer: str,
    context: str
) -> str:
    """
    Generates a guiding question to help the student.
    Args:
        llm: The LLM instance (e.g., ChatGoogleGenerativeAI).
        user_language: 'en' or 'he'.
        chat_history: List of previous messages.
        question: The current math question.
        answer: Student's answer.
        context: Retrieved context for the question.
    Returns:
        A guiding question string.
    """
    try:
        guiding_chain = get_guiding_question_prompt(user_language) | llm
        response = guiding_chain.invoke({
            "chat_history": chat_history[-3:],
            "question": question,
            "answer": answer,
            "context": context
        })
        return response.content.strip()
    except Exception as e:
        print(f"Error generating guiding question: {e}")
        return f"{'What do you think the first step should be?' if user_language == 'en' else 'מה אתה חושב שצריך להיות הצעד הראשון?'}"

def describe_svg_content(llm, svg_content: str) -> str:
    """
    Describes the mathematical elements in SVG content.
    Args:
        llm: The LLM instance.
        svg_content: The SVG content string.
    Returns:
        A concise description of the SVG's mathematical elements.
    """
    try:
        svg_description_chain = get_svg_description_prompt() | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return response.content.strip()
    except Exception as e:
        print(f"Error describing SVG content: {e}")
        return "An error occurred while describing the image."

def translate_text_to_english(llm, text: str) -> str:
    """
    Translates text to English if needed.
    Args:
        llm: The LLM instance.
        text: Input text to translate.
    Returns:
        Translated text (or original if already English).
    """
    if not text or not text.strip():
        return text
    try:
        translation_chain = get_translation_prompt() | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = response.content.strip()
        if is_likely_hebrew(text) and not is_likely_hebrew(translated):
            return translated
        elif not is_likely_hebrew(text):
            return text
        else:
            print(f"Potential translation issue. Input: {text}, Output: {translated}")
            return translated
    except Exception as e:
        print(f"Error translating text: {e}")
        return f"[Translation Error: {text}]"

def is_likely_hebrew(text: str) -> bool:
    """
    Checks if text contains Hebrew characters.
    Args:
        text: Input text.
    Returns:
        True if Hebrew characters are detected, False otherwise.
    """
    return any('\u0590' <= char <= '\u05FF' for char in text)

def generate_humor_response(llm, user_language: str, user_input: str) -> str:
    """
    Generates a humorous response to match user's laughter.
    Args:
        llm: The LLM instance.
        user_language: 'en' or 'he'.
        user_input: User's input containing humor indicators.
    Returns:
        A short humorous response or emoji.
    """
    try:
        humor_chain = get_humor_response_prompt(user_language) | llm
        response = humor_chain.invoke({"input": user_input})
        return response.content.strip()
    except Exception as e:
        print(f"Error generating humor response: {e}")
        return "Haha, that's funny! 😂" if user_language == "en" else "חה חה, זה מצחיק! 😂"

def generate_solution_explanation(
    llm,
    user_language: str,
    chat_history: List,
    question: str,
    solution: str
) -> str:
    """
    Generates a step-by-step explanation for a solution.
    Args:
        llm: The LLM instance.
        user_language: 'en' or 'he'.
        chat_history: List of previous messages.
        question: The math question.
        solution: The correct solution.
    Returns:
        A detailed explanation string.
    """
    try:
        explanation_chain = get_solution_explanation_prompt(user_language) | llm
        response = explanation_chain.invoke({
            "chat_history": chat_history[-3:],
            "question": question,
            "solution": solution
        })
        return response.content.strip()
    except Exception as e:
        print(f"Error generating solution explanation: {e}")
        return f"Solution: {solution}"  # Fallback

def generate_lesson_summary(
    llm,
    user_language: str,
    chat_history: List,
    diagnostic: dict
) -> str:
    """
    Generates a concise lesson summary.
    Args:
        llm: The LLM instance.
        user_language: 'en' or 'he'.
        chat_history: List of previous messages.
        diagnostic: Dictionary of diagnostic answers.
    Returns:
        A 2-3 sentence summary.
    """
    try:
        summary_chain = get_summary_prompt(user_language) | llm
        response = summary_chain.invoke({
            "chat_history": chat_history[-10:],
            "diagnostic": json.dumps(diagnostic)
        })
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Great lesson today!" if user_language == "en" else "שיעור נהדר היום!"