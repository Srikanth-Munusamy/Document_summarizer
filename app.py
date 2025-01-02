import streamlit as st
from helper_functions import (
    extract_text_from_pdf,
    split_text_into_chunks,
    create_pinecone_index,
    retrieve_chunks,
)
from groq_summarizer import summarize_text  # Importing the Groq summarizer

# App title
st.title("Legal Document Summarization & Chatbot")
st.sidebar.title("Upload & Process Documents")

# Sidebar file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Initialize variables
index_name = "legal-docs-index"

if uploaded_file:
    # Extract text from the uploaded PDF
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("### Uploaded Document Text:")
    st.write(document_text[:2000])  # Display first 2000 characters

    # Summarize document using Groq
    if st.button("Summarize Document"):
        summary = summarize_text(document_text)  # Summarizing using Groq API
        if summary:
            st.write("### Summary:")
            st.write(summary)
        else:
            st.write("Error summarizing the document")

    # Ask questions using RAG
    st.write("### Ask Questions:")
    if "vector_store" not in st.session_state:
        chunks = split_text_into_chunks(document_text)
        st.session_state.vector_store = create_pinecone_index(index_name, chunks)

    user_query = st.text_input("Ask a question about the document:")
    if user_query:
        relevant_text = retrieve_chunks(user_query, st.session_state.vector_store)
        response = summarize_text(relevant_text + f"\n\nAnswer the question: {user_query}")  # Summarizing relevant text for the question
        if response:
            st.write("### Response:")
            st.write(response)
        else:
            st.write("Error generating the response.")
