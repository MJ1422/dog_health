from flask import Flask, request, jsonify
from flask_cors import CORS
from cohere import Client

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Cohere Client with your API key
co = Client('tvIZ363Cg95OtTsniNW9YoglrZ6ApbdHwpalpsdT')

# New Cohere Model ID
MODEL_ID = 'a49b62f0-3c51-47a0-bda7-aa26df546bfb-ft'

# Store conversation context
user_context = {}

def classify_text(question):
    try:
        response = co.classify(model=MODEL_ID, inputs=[question])
        return response.classifications[0].prediction
    except Exception as e:
        print(f"Classification Error: {e}")
        return None

def ask_chatbot(user_id, question):
    category = classify_text(question)

    # Allow general greetings and polite responses
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    polite_responses = ["thank you", "thanks"]

    if any(word in question.lower() for word in greetings):
        return "Hello! How can I assist you with your dog's concerns?"
    elif any(word in question.lower() for word in polite_responses):
        return "You're welcome! Let me know if you need any help with your dog."
    
    # Restrict to dog-related topics based on classification
    if category != "dog topic":
        return "I can only assist with dog-related topics."

    if user_id not in user_context:
        user_context[user_id] = []
    
    user_context[user_id].append(question)

    try:
        response = co.chat(model='command', message=question)
        return response.text.strip().replace("**", "")
    except Exception as e:
        print(f"Chatbot API Error: {e}")
        return "I'm sorry, but I couldn't process your request right now."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default")
    
    if not user_message:
        return jsonify({"response": "Please enter a valid message."})
    
    chatbot_response = ask_chatbot(user_id, user_message)
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(debug=True)
