import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from modules.reminder import generate_daily_checklist
from ui.styles import apply_custom_style
apply_custom_style()

st.set_page_config(page_title="Daily Reminders", page_icon="⏰")
st.title("Daily Reminders")

if "medicines" not in st.session_state:
    st.session_state.medicines = []

if not st.session_state.medicines:
    st.info("Upload a prescription first (see 'Upload Prescription' page) to generate a checklist.")
else:
    if st.button("Generate today's checklist"):
        checklist = generate_daily_checklist(st.session_state.medicines)
        st.markdown(checklist)