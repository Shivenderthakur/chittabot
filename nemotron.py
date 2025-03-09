import os
import requests
import logging
import asyncio  # <-- Add this

from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Retrieve API keys from environment variables
NEMETRON_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NEMETRON_API_KEY:
    logging.error("API key for Nemetron is missing.")
    raise ValueError("API key for Nemetron is required.")

logging.debug("API key loaded successfully.")

# Initialize Nemetron Client
client = ChatNVIDIA(
    model="nvidia/nemotron-4-340b-instruct",
    api_key=NEMETRON_API_KEY,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)
logging.debug("Nemetron API client initialized.")

# Function to query Nemetron API
async def query_nemetron_llm(gemini_results, user_query):
    logging.debug(f"Processing Gemini API results for Nemetron query: {gemini_results}")

    gemini_content = gemini_results.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    if not gemini_content.strip():
        logging.warning("Gemini API returned an empty response.")
        return "I'm sorry, I couldn't find relevant information."

    # Detect if the user query is related to mental health
    mental_health_keywords = ["depression", "anxiety", "stress", "mental health", "therapy", "counseling", "psychologist", "suicide", "self-harm", "mindfulness"]
    
    if any(keyword in user_query.lower() for keyword in mental_health_keywords):
        prompt = f"""You are a **mental health advisory chatbot** focused on helping users with mental well-being.
        Provide compassionate, well-informed, and non-judgmental responses based on Indian mental health resources.
        Answer the following query with a supportive tone:
        {user_query}

        Use the following content for additional insights:
        {gemini_content}"""
    else:
        prompt = f"""You are a general AI assistant, similar to ChatGPT.
        Respond informatively and concisely to user queries across various topics.
        Here is the user's question:
        {user_query}"""

    logging.debug(f"Nemetron prompt: {prompt}")

    try:
        response = await asyncio.to_thread(client.stream, [{"role": "user", "content": prompt+"here is the content related data if usefule and use it"}])
        result = "".join(chunk.content for chunk in response)
        logging.debug("Nemetron response received successfully.")
        return result
    except Exception as e:
        logging.exception("Error querying Nemetron API:")
        return "Error fetching insights from Nemetron API."
