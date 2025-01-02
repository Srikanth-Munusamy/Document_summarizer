import requests
import os

# Load the Groq API Key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def summarize_text(text):
    url = "https://api.groq.com/openai/v1/chat/completions"  # Groq API endpoint

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Create the message format expected by the Groq API for a chat model
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this legal document:\n\n{text}"}
        ],
        "model": "llama3-8b-8192",  # Ensure you are using the correct model
        "max_tokens": 150  # Adjust the tokens based on your needs
    }

    # Make the API request
    response = requests.post(url, json=data, headers=headers)

    # Handle the response
    if response.status_code == 200:
        # Assuming the response contains 'choices' with the 'message' object that has the summarized text
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        # Handle error response
        return f"Error: {response.status_code}, {response.text}"
