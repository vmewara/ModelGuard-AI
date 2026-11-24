
import streamlit as st
import sys

sys.path.append("/content/ModelGuard_AI")

from genai.modelguard_engine import ModelGuardEngine

st.set_page_config(
    page_title="ModelGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ModelGuard AI")
st.caption("Explainable ML Monitoring and AI-Powered Investigation")

# Project status
col1, col2, col3 = st.columns(3)

col1.metric("Features monitored", "4")
col2.metric("High PSI features", "2")
col3.metric("AI capabilities", "3")

st.divider()

# AI feature selection
st.subheader("AI Monitoring Assistant")

feature = st.selectbox(
    "Select an AI capability",
    [
        "Monitoring Assistant",
        "Incident Investigator",
        "Performance Advisor"
    ]
)

feature_mapping = {
    "Monitoring Assistant": "assistant",
    "Incident Investigator": "incident",
    "Performance Advisor": "performance"
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("Ask ModelGuard AI a question...")

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("ModelGuard AI is analysing your request..."):

            try:
                response = engine.ask(
                    question=question,
                    feature=feature_mapping[feature]
                )

                st.markdown(response)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Error: {e}")
                
