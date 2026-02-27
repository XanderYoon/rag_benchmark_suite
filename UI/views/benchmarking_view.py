from __future__ import annotations

import streamlit as st


def render(show_title: bool = True) -> None:
    if show_title:
        st.title("Benchmarking")
    else:
        st.subheader("Benchmarking")
    st.info("Blank for now.")
