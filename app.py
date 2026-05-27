from flask import Flask, render_template, request, jsonify

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

from dotenv import load_dotenv

import os

# Load env
load_dotenv()

app = Flask(__name__)

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medicalbot")

# Embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Chat memory
chat_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():

    user_message = request.form["msg"]

    # Save user message
    chat_history.append(f"User: {user_message}")

    # Convert query to embedding
    query_embedding = embedding_model.encode(user_message).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Build context
    context = ""

    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n"

    # Memory context
    memory = "\n".join(chat_history[-5:])

    # Prompt
    prompt = f"""
    You are MediMind, a helpful medical AI assistant.

    Previous Conversation:
    {memory}

    Medical Context:
    {context}

    User Question:
    {user_message}

    Rules:
    - Answer clearly
    - Use medical context
    - Be friendly
    - Do not prescribe dangerous medication
    - Suggest doctor consultation if serious
    """

    # LLM response
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content

    # Save AI response
    chat_history.append(f"AI: {answer}")

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)