from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="RAG Benchmark Builder", layout="wide")

pages = [
    st.Page("UI/pages/1_ingest.py", title="Ingest"),
    st.Page("UI/pages/2_question_generation.py", title="Question Generation"),
    st.Page("UI/pages/3_verify_questions.py", title="Verify Questions"),
]

navigation = st.navigation(pages)
navigation.run()
