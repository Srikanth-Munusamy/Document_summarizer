import pdfplumber
from dotenv import load_dotenv
import os
import requests  # Import for API calls to Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec  # Correct Pinecone initialization
import numpy as np  # Needed for handling embeddings

# Load environment variables
load_dotenv()

# Load API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")  # Assuming the Groq API key is stored in .env
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# PDF Parsing
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Text Summarization (Updated API Call for Groq)
def summarize_text(text):
    url = "https://api.groq.ai/v1/summarize"  # Replace with the actual Groq API URL for summarization
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model": "groq-gpt-4",  # Adjust the model name if needed
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        summary = response.json().get('summary', '')
        return summary.strip() if summary else "No summary available."
    except requests.exceptions.RequestException as e:
        print(f"Error summarizing text: {e}")
        return "Error generating summary."

# Split Text into Chunks
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)

# Create Pinecone Index (Fixing from_texts issue)
def create_pinecone_index(index_name, chunks):
    # Remove OpenAIEmbeddings since you're not using OpenAI
    embeddings = None  # Placeholder for your embeddings (use custom embeddings or skip for now)

    # Check if the index exists, otherwise create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ensure dimension matches the embedding size if you add embeddings later
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Use Pinecone's correct method to access the index
    index = pc.Index(index_name)

    # For now, manually create embeddings (if any)
    # If using a custom embedding function, replace `embeddings` with actual embedding data
    embeddings_data = np.random.rand(len(chunks), 1536)  # Example: Random embeddings (Replace with real embeddings)

    # Upsert the embeddings and the corresponding text into Pinecone
    vectors = [(str(i), embedding.tolist(), {"text": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_data))]

    index.upsert(vectors)

    return index

# Retrieve Relevant Chunks (Updated query method with keyword arguments)
def retrieve_chunks(query, vector_store):
    # Assuming you have a query embedding for similarity search
    query_embedding = np.random.rand(1, 1536)  # Example: Random query embedding (Replace with actual query embedding)

    # Updated query method with keyword arguments
    results = vector_store.query(vector=query_embedding.tolist(), top_k=3)

    return " ".join([match['metadata']['text'] for match in results['matches']])
