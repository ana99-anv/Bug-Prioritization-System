
import streamlit as st

st.set_page_config(
    page_title="Bug Prioritization System",
    page_icon="🐞",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🐞 Intelligent Bug Prioritization System")

st.markdown("""
## AI-Assisted Bug Triage

This system uses machine learning to:

- Predict bug priority from **P1–P5**
- Identify potentially **urgent bugs**
- Provide probability-based recommendations
- Support **human-in-the-loop** triage
""")

st.divider()

# ============================================================
# KEY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Multiclass Accuracy",
        "66.75%"
    )

with col2:
    st.metric(
        "Multiclass Macro-F1",
        "0.326"
    )

with col3:
    st.metric(
        "Binary Accuracy",
        "85.90%"
    )

with col4:
    st.metric(
        "Urgent Recall",
        "80.07%"
    )

st.divider()

# ============================================================
# WORKFLOW
# ============================================================

st.subheader("🔄 System Workflow")

st.markdown("""
**Bug Report**

↓

**Feature Extraction**

↓

**Priority Prediction**

↓

**Urgent Probability**

↓

**Threshold Decision**

↓

**Human Review**
""")

st.divider()

# ============================================================
# THRESHOLD EXPLANATION
# ============================================================

st.subheader("🎯 Threshold-Based Urgent Screening")

st.markdown("""
The binary model produces an **urgent probability**.

The probability threshold determines whether the bug is sent
to the urgent-review queue.

A lower threshold:

- catches more urgent bugs
- increases recall
- increases human-review workload

A higher threshold:

- reduces unnecessary reviews
- increases precision
- risks missing more urgent bugs
""")

st.info("""
**Human-in-the-loop:** The model provides a recommendation.
A human triager makes the final decision.
""")

st.divider()

# ============================================================
# NAVIGATION
# ============================================================

st.markdown("""
### Use the sidebar

📊 **Dashboard**
View model performance and product metrics.

🔮 **Predict Bug**
Predict a single bug.

📂 **Batch Prediction**
Predict multiple bugs from a CSV.

🔍 **Model Insights**
Explore threshold trade-offs, cost analysis and model behavior.
""")