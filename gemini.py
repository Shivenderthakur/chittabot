import os
import requests
import logging
import asyncio  # <-- Add this

# Retrieve API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("API key for Gemini is missing.")
    raise ValueError("API key for Gemini is required.")

logging.debug("API key loaded successfully.")

# Gemini API Configuration
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

async def query_gemini_api(query):
    logging.debug(f"Querying Gemini API with: {query}")
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": query}]}]}

    try:
        response = await asyncio.to_thread(requests.post, GEMINI_API_URL, headers=headers, json=payload)
        logging.debug(f"Gemini API response status: {response.status_code}")

        json_response = response.json()
        logging.debug(f"Full Gemini API Response: {json_response}")

        if response.status_code == 200:
            return json_response
        
        else:
            logging.error(f"Gemini API request failed: {response.text}")
            return {"error": f"Gemini API failed: {response.status_code}"}
        print(json_response)
    except Exception as e:
        logging.exception("Error querying Gemini API:")
        return {"error": str(e)}
