from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import cohere

app = Flask(__name__)
CORS(app)

# Initialize Cohere Client with your API key
co = cohere.Client('tvIZ363Cg95OtTsniNW9YoglrZ6ApbdHwpalpsdT')

# New Cohere Model ID
MODEL_ID = 'a49b62f0-3c51-47a0-bda7-aa26df546bfb-ft'

# Store conversation context
user_context = {}

# Caching results to avoid redundant API calls
cache = {}

async def classify_text(question):
    """Classify the text using Cohere API and cache the result."""
    if question in cache:
        return cache[question]  # Return cached classification

    try:
        response = await asyncio.to_thread(co.classify, model=MODEL_ID, inputs=[question])
        category = response.classifications[0].prediction
        cache[question] = category  # Cache the result
        return category
    except Exception as e:
        print(f"Classification Error: {e}")
        return None

async def ask_chatbot(user_id, question):
    """Process user question and get chatbot response."""
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    polite_responses = ["thank you", "thanks"]

    if any(word in question.lower() for word in greetings):
        return "Hello! How can I assist you with your dog's concerns?"
    elif any(word in question.lower() for word in polite_responses):
        return "You're welcome! Let me know if you need any help with your dog."

    # Classify text asynchronously
    category = await classify_text(question)

    # Restrict to dog-related topics
    if category != "dog topic":
        return "I can only assist with dog-related topics."

    # Store user conversation context
    user_context.setdefault(user_id, []).append(question)

    # Get chatbot response asynchronously
    try:
        response = await asyncio.to_thread(co.chat, model='command', message=question)
        return response.text.strip().replace("**", "")
    except Exception as e:
        print(f"Chatbot API Error: {e}")
        return "I'm sorry, but I couldn't process your request right now."

@app.route("/chat", methods=["POST"])
async def chat():
    """Handle chat requests asynchronously."""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default")

    if not user_message:
        return jsonify({"response": "Please enter a valid message."})

    chatbot_response = await ask_chatbot(user_id, user_message)
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(debug=True)
