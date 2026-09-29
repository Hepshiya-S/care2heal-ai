import streamlit as st


def apply_custom_style():
    st.markdown("""
        <style>
        .card {
            background-color: white;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(124, 92, 252, 0.08);
        }
        .greeting {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2D2A3D;
        }
        div.stButton > button {
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)