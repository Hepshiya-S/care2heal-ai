import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from ui.styles import apply_custom_style
from agents.agent_graph import app as agent_app
from modules.voice_io import record_audio, transcribe_and_translate_to_english, translate_and_speak, translate_to_english

st.set_page_config(page_title="Ask Care2Heal", page_icon="💬")
apply_custom_style()
st.title("Ask Care2Heal")

LANGUAGES = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
lang_name = st.selectbox("Response language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[lang_name]

record_seconds = st.slider("Recording length (seconds)", 5, 15, 8)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_answer(question_in_english: str, lang_code: str, lang_name: str):
    result = agent_app.invoke({
        "question": question_in_english,
        "documents": [],
        "metadatas": [],
        "relevant": False,
        "is_emergency": False,
        "is_smalltalk": False,
        "answer": ""
    })
    answer = result["answer"]
    audio_path = None
    if lang_code != "en":
        audio_path = f"response_{int(time.time() * 1000)}.mp3"
        answer = translate_and_speak(answer, lang_code, lang_name, output_path=audio_path)
    return answer, audio_path


def add_message_and_respond(displayed_text: str, question_for_agent: str):
    st.session_state.chat_history.append({"role": "user", "content": displayed_text})
    answer, audio_path = get_answer(question_for_agent, lang_code, lang_name)
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "audio": audio_path})


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("audio") and os.path.exists(msg["audio"]):
            st.audio(msg["audio"])

col1, col2 = st.columns([5, 1])
with col1:
    typed_question = st.chat_input("Type your question")
with col2:
    mic_clicked = st.button("🎤", help="Record a question")

if typed_question:
    question_for_agent = typed_question
    if lang_code != "en":
        question_for_agent = translate_to_english(typed_question)
    add_message_and_respond(typed_question, question_for_agent)
    st.rerun()

if mic_clicked:
    with st.spinner(f"Recording for {record_seconds} seconds..."):
        audio_path = record_audio(duration=record_seconds)
        spoken_question = transcribe_and_translate_to_english(audio_path)
    add_message_and_respond(spoken_question, spoken_question)
    st.rerun()