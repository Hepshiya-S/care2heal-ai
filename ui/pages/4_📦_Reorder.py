import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from modules.reorder import generate_reorder_qr, confirm_and_reorder
from ui.styles import apply_custom_style
apply_custom_style()

st.set_page_config(page_title="Reorder", page_icon="📦")
st.title("Reorder Medicine")

if "medicines" not in st.session_state:
    st.session_state.medicines = []

if not st.session_state.medicines:
    st.info("Upload a prescription first (see 'Upload Prescription' page) to reorder.")
else:
    med_names = [m["name"] for m in st.session_state.medicines]
    chosen = st.selectbox("Select medicine", med_names)
    if st.button("Generate reorder QR"):
        qr_path = generate_reorder_qr(chosen, f"https://www.1mg.com/search/all?name={chosen}")
        st.image(qr_path, caption=f"Scan to reorder {chosen}")
    if st.button("Confirm reorder"):
        st.success(confirm_and_reorder(chosen))