
import streamlit as st

st.set_page_config(
    page_title="Results",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a page below.")

st.markdown(
    """
    Select Predictions to see plot for predictions from various models.
    Select Errors to compare overall errors among various models
    """
)