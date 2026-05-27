from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("medicalbot")

# Embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# User query
query = input("Ask Medical Question: ")

# Convert query to embedding
query_embedding = embedding_model.encode(query).tolist()

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# Extract context
context = ""

for match in results["matches"]:
    context += match["metadata"]["text"] + "\n"

print("\nRetrieved Context:\n")
print(context)

# AI Prompt
prompt = f"""
You are a helpful medical assistant.

Use ONLY the provided medical context.

If medicines are mentioned in the context,
list them clearly.

Do not say "I could not find this in the medical book"
if relevant information exists.

Context:
{context}

Question:
{query}

Give a clear and short answer.
"""
# Ask Groq
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
)

# Print answer
answer = chat_completion.choices[0].message.content

print("\nAI Answer:\n")
print(answer)