import streamlit as st
import logging
import asyncio
from nemotron import query_nemetron_llm
from gemini import query_gemini_api

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("Mental Health ")

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am CHINTACHODO "}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("Ask about mental health resources in India..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Asynchronously fetch Gemini API results
        gemini_results = asyncio.run(query_gemini_api(prompt))
        logging.debug("Gemini API results obtained.")

        # Asynchronously generate insights from Nemetron LLM
        insights = asyncio.run(query_nemetron_llm(gemini_results,prompt))
        logging.debug("Insights generated from Nemetron API.")

        full_response = insights
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
