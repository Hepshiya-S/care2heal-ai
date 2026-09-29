import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.styles import apply_custom_style

st.set_page_config(page_title="Care2Heal", page_icon="💊", layout="centered")
apply_custom_style()

if "medicines" not in st.session_state:
    st.session_state.medicines = []

st.markdown('<div class="greeting">How are you feeling today? 💜</div>', unsafe_allow_html=True)
st.caption("Your Care2Heal assistant is here to help")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("📄 **Upload Prescription** — scan a photo to check your medicines")
st.write("💬 **Ask Care2Heal** — type or speak any question")
st.write("⏰ **Daily Reminders** — see today's medicine checklist")
st.write("📦 **Reorder** — scan a QR or confirm a refill")
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.medicines:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Your medicines")
    for med in st.session_state.medicines:
        st.write(f"💊 **{med['name']}** {med['dosage']} — {med['instructions']}")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No medicines on file yet — start with 'Upload Prescription' in the sidebar.")