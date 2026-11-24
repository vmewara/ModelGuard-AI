
import os
import streamlit as st
import pandas as pd
from groq import Groq

from genai.modelguard_engine import ModelGuardEngine


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ModelGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# 2. GROQ API CONFIGURATION
# --------------------------------------------------

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing. Configure it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)


# --------------------------------------------------
# 3. MONITORING EVIDENCE
# --------------------------------------------------

psi_report = pd.DataFrame([
    {"Feature": "MonthlyCharges", "PSI": 1.8121, "Status": "High"},
    {"Feature": "tenure", "PSI": 1.5449, "Status": "High"},
    {"Feature": "SeniorCitizen", "PSI": 0.0000, "Status": "Low"},
    {"Feature": "TotalCharges", "PSI": 0.0000, "Status": "Low"}
])

ks_report = pd.DataFrame([
    {
        "Feature": "MonthlyCharges",
        "KS Statistic": 0.2434,
        "p-value": 0.0,
        "Possible Drift": True
    },
    {
        "Feature": "tenure",
        "KS Statistic": 0.2250,
        "p-value": 0.0,
        "Possible Drift": True
    },
    {
        "Feature": "SeniorCitizen",
        "KS Statistic": 0.0000,
        "p-value": 1.0,
        "Possible Drift": False
    },
    {
        "Feature": "TotalCharges",
        "KS Statistic": 0.0000,
        "p-value": 1.0,
        "Possible Drift": False
    }
])

alerts_df = pd.DataFrame([
    {"Feature": "MonthlyCharges", "Alert": "High PSI drift"},
    {"Feature": "tenure", "Alert": "High PSI drift"},
    {"Feature": "MonthlyCharges", "Alert": "KS drift detected"},
    {"Feature": "tenure", "Alert": "KS drift detected"}
])


# --------------------------------------------------
# 4. INITIALIZE MODELGUARD ENGINE
# --------------------------------------------------

engine = ModelGuardEngine(
    client=client,
    psi_report=psi_report,
    ks_report=ks_report,
    alerts_df=alerts_df
)


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("🛡️ ModelGuard AI")

st.caption(
    "Explainable ML Monitoring | GenAI Investigation | "
    "Model Performance Advisor"
)

st.divider()


# --------------------------------------------------
# 6. DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Features Monitored",
    "4"
)

col2.metric(
    "High PSI Features",
    "2",
    delta="Drift detected"
)

col3.metric(
    "AI Capabilities",
    "3"
)

st.divider()


# --------------------------------------------------
# 7. PSI MONITORING REPORT
# --------------------------------------------------

st.subheader("📊 Feature Drift Monitoring")

st.dataframe(
    psi_report,
    use_container_width=True,
    hide_index=True
)

st.subheader("📈 KS Distribution Test")

st.dataframe(
    ks_report,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Experiment note: The current dataset was simulated by "
    "increasing MonthlyCharges by 15 and multiplying tenure "
    "by 0.8. These findings are not confirmed production incidents."
)

st.divider()


# --------------------------------------------------
# 8. SELECT AI CAPABILITY
# --------------------------------------------------

st.subheader("🤖 ModelGuard AI Assistant")

feature = st.selectbox(
    "Choose an AI capability",
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

st.caption(
    "Ask questions about monitoring results, drift investigation "
    "or model performance."
)


# --------------------------------------------------
# 9. CHAT HISTORY
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])


# --------------------------------------------------
# 10. USER QUESTION AND AI RESPONSE
# --------------------------------------------------

question = st.chat_input(
    "Ask ModelGuard AI a question..."
)

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

                error_message = str(e)

                if "429" in error_message or "rate_limit" in error_message.lower():
                    st.warning(
                        "Groq API rate limit reached. "
                        "Please wait a little and try again."
                    )
                else:
                    st.error(f"ModelGuard AI Error: {error_message}")


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ModelGuard AI | GenAI-powered ML monitoring prototype"
)
