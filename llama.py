import streamlit as st
import os
import asyncio
import requests
import logging
import speech_recognition as sr
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from streamlit_webrtc import webrtc_streamer

# Set API Keys
NVIDIA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not NVIDIA_API_KEY or not GEMINI_API_KEY:
    st.error("Missing API keys. Set NVIDIA_API_KEY and GEMINI_API_KEY in your environment.")
    st.stop()

# Initialize NVIDIA LangChain client (Llama 3)
client = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=NVIDIA_API_KEY,
    temperature=0.2,
    top_p=0.7,
    max_tokens=512,
)

# Gemini API Configuration
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Query Gemini API for mental health resources
async def query_gemini_api(query):
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": query}]}]}

    try:
        response = await asyncio.to_thread(requests.post, GEMINI_API_URL, headers=headers, json=payload)
        json_response = response.json()
        
        if response.status_code == 200:
            return json_response
        else:
            logging.error(f"Gemini API request failed: {response.text}")
            return {"error": "Failed to fetch resources."}
    except Exception as e:
        logging.exception("Error querying Gemini API:")
        return {"error": str(e)}

# Streamlit UI
st.title("ChittaBot")
st.text("Chitta ko shanti, takneek se sahara")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle voice input
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=10)
            text = r.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand that. Please try again.")
        except sr.RequestError:
            st.error("Could not request results, please check your connection.")

# Query NVIDIA API with context-aware response
async def query_llama_stream(prompt, chat_history):
    # Fetch mental health resources if the query is related
    gemini_results = await query_gemini_api(prompt)
    resources = gemini_results.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    # Modify Llama's prompt dynamically
    final_prompt = f"you are a professional mental health counsellor ai called as ChittaBot developed by the CodeWarriors .Provide a helpful yet concise response with some details for mental health if this consist of simple conversation go with the flow  based on this query: {prompt}.\nIf useful, mention resources with some detail and if URLs exist, include them:\n{resources}"


    messages = chat_history + [{"role": "user", "content": final_prompt}]
    stream = client.stream(messages)

    full_response = ""
    for chunk in stream:
        if chunk.content:
            full_response += chunk.content
            yield chunk.content  # Streaming response in real-time

    # Save response to memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Handle user input
col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.chat_input("Ask about mental health or anything else...")

with col2:
    if st.button("🎙️ Speak"):
        prompt = recognize_speech()

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_generator = query_llama_stream(prompt, st.session_state.messages)
        st.write_stream(response_generator)
