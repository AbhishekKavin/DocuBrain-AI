import streamlit as st
import requests
import json

st.set_page_config(page_title="DocuBrain AI", layout = "wide")
st.title("DocuBrain: Document Intelligence")

with st.sidebar:
    st.header("System Status")
    if st.button("Check health"):
        health = requests.get("http://127.0.0.1:8000/health").json()
        st.write(health)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({'role':'user', "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        url = "http://127.0.0.1:8000/ask-stream"
        with requests.post(url, json = {"question": prompt}, stream = True) as r:
            for chunk in r.iter_content(chunk_size = None):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    if decoded_chunk.startswith("SOURCES:"):
                        st.caption(f"Sources: {decoded_chunk.replace('SOURCES: ','')}")
                    else:
                        full_response += decoded_chunk
                        response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
    st.session_state.messages.append({'role':'assistant','content':full_response})    