# Import necessary components from the LangChain and Python libraries.
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. SET UP THE LLM AND API KEY ---
# Replace 'your-api-key' with your actual Google API key.
# It's best practice to use environment variables for API keys.
os.environ["GOOGLE_API_KEY"] = "AIzaSyAdKYyRoN1R7G9KERy2HuQZ2Pabs4mSOkY"

# Initialize the LLM. We'll use the ChatGoogleGenerativeAI model with 'gemini-pro'.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- 2. INITIALIZE CHAT HISTORY ---
# We'll use a simple Python list to store the chat history.
chat_history = []

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

# --- 5. START THE CHAT LOOP ---
print("A_GUY the math tutor is ready. Type 'exit' to end the chat.")

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

    # We want to only send the last 5-6 messages to the LLM to maintain context.+
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