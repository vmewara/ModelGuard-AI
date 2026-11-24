
import os
import streamlit as st
import pandas as pd
from groq import Groq

from genai.modelguard_engine import ModelGuardEngine


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ModelGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# 2. GROQ CONFIGURATION
# ==================================================

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key is missing. Configure GROQ_API_KEY in Secrets.")
    st.stop()

client = Groq(api_key=api_key)


# ==================================================
# 3. MONITORING DATA
# ==================================================

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


# ==================================================
# 4. MODEL PERFORMANCE DATA
# ==================================================

model_metrics = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Random Forest": 0.7786,
        "ANN": 0.7871
    },
    {
        "Metric": "Precision",
        "Random Forest": 0.5721,
        "ANN": 0.5944
    },
    {
        "Metric": "Recall",
        "Random Forest": 0.6578,
        "ANN": 0.6230
    },
    {
        "Metric": "F1-Score",
        "Random Forest": 0.6119,
        "ANN": 0.6084
    },
    {
        "Metric": "ROC-AUC",
        "Random Forest": 0.8346,
        "ANN": 0.8397
    }
])


# ==================================================
# 5. INITIALIZE MODELGUARD ENGINE
# ==================================================

engine = ModelGuardEngine(
    client=client,
    psi_report=psi_report,
    ks_report=ks_report,
    alerts_df=alerts_df
)


# ==================================================
# 6. HEADER
# ==================================================

st.title("🛡️ ModelGuard AI")

st.caption(
    "Explainable ML Monitoring | "
    "GenAI Investigation | "
    "Model Performance Advisor"
)

st.divider()


# ==================================================
# 7. DASHBOARD METRICS
# ==================================================

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


# ==================================================
# 8. FEATURE DRIFT MONITORING
# ==================================================

st.header("📊 Feature Drift Monitoring")

left, right = st.columns(2)

with left:
    st.subheader("Population Stability Index")
    st.table(psi_report)

with right:
    st.subheader("Kolmogorov-Smirnov Test")
    st.table(ks_report)

st.warning(
    "Simulation notice: MonthlyCharges was increased by 15 "
    "and tenure was multiplied by 0.8. "
    "These are simulated drift findings, not confirmed "
    "production incidents."
)

st.divider()


# ==================================================
# 9. MODEL COMPARISON
# ==================================================

st.header("📈 Model Performance Comparison")

st.dataframe(
    model_metrics.style.format({
        "Random Forest": "{:.4f}",
        "ANN": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Metrics are from the existing ModelGuard experiment."
)

st.divider()


# ==================================================
# 10. AI CAPABILITIES
# ==================================================

st.header("🤖 ModelGuard AI Assistant")

feature = st.selectbox(
    "Choose your AI assistant",
    [
        "General ModelGuard Assistant",
        "Monitoring Assistant",
        "Incident Investigator",
        "Performance Advisor"
    ]
)

feature_mapping = {
    "General ModelGuard Assistant": "assistant",
    "Monitoring Assistant": "assistant",
    "Incident Investigator": "incident",
    "Performance Advisor": "performance"
}

descriptions = {
    "General ModelGuard Assistant":
        "Ask questions about ML monitoring, drift, models and the available evidence.",

    "Monitoring Assistant":
        "Understand PSI, KS tests, feature drift and monitoring alerts.",

    "Incident Investigator":
        "Investigate detected drift and receive practical engineering recommendations.",

    "Performance Advisor":
        "Compare model metrics and understand performance trade-offs."
}

st.info(descriptions[feature])


# ==================================================
# 11. CHAT HISTORY
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])


# ==================================================
# 12. ASK QUESTIONS
# ==================================================

question = st.chat_input(
    "Ask your own question to ModelGuard AI..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("ModelGuard AI is analysing..."):

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

                error = str(e)

                if "429" in error or "rate_limit" in error.lower():

                    st.warning(
                        "Groq rate limit reached. "
                        "Wait a few seconds and try again."
                    )

                else:

                    st.error(
                        f"ModelGuard AI Error: {error}"
                    )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "ModelGuard AI | GenAI-powered ML monitoring prototype"
)
