import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from modules.ocr_reader import extract_text_from_image, parse_prescription_text
from modules.safety_checker import check_drug_interactions
from ui.styles import apply_custom_style
apply_custom_style()

st.set_page_config(page_title="Upload Prescription", page_icon="📄")
st.title("Upload Prescription")

if "medicines" not in st.session_state:
    st.session_state.medicines = []

uploaded_file = st.file_uploader("Upload prescription/discharge photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp_upload.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    raw_text = extract_text_from_image("temp_upload.jpg")
    st.session_state.medicines = parse_prescription_text(raw_text)

if st.session_state.medicines:
    st.subheader("Extracted medicines")
    for med in st.session_state.medicines:
        st.write(f"- **{med['name']}** {med['dosage']} — {med['instructions']}")

    if st.button("Check interactions"):
        names = [m["name"] for m in st.session_state.medicines]
        warnings = check_drug_interactions(names)
        if warnings:
            for w in warnings:
                st.error(w)
        else:
            st.success("No known interactions found.")