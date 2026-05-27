from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

import os
import uuid

# Load environment variables
load_dotenv()

# Load PDF
loader = PyPDFLoader("data/medical_book.pdf")
docs = loader.load()

print(f"Total Pages: {len(docs)}")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(docs)

print(f"Total Chunks: {len(texts)}")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Connect index
index = pc.Index(index_name)

# Prepare vectors
vectors = []

for text in texts:
    embedding = embeddings.embed_query(text.page_content)

    vectors.append({
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "text": text.page_content
        }
    })

print("Uploading vectors to Pinecone...")

# Upload in batches
batch_size = 100

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)

    print(f"Uploaded {i + len(batch)} vectors")

print("Data uploaded successfully!")