import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bug Prioritization System",
    page_icon="🐞",
    layout="wide"
)


# ============================================================
# PROJECT METRICS
# ============================================================

MULTICLASS_ACCURACY = 0.6675
MULTICLASS_MACRO_F1 = 0.3256
BINARY_ACCURACY = 0.8590

DEFAULT_THRESHOLD = 0.50
RECALL_THRESHOLD = 0.097
COST_THRESHOLD = 0.08

DEFAULT_PRECISION = 0.571
DEFAULT_RECALL = 0.1237
DEFAULT_F1 = 0.2018

RECALL_TARGET_PRECISION = 0.2435
RECALL_TARGET_RECALL = 0.8007

COST_PRECISION = 0.2171
COST_RECALL = 0.8729

FN_COST = 10
FP_COST = 1

TEMPORAL_ACCURACY = 0.6170
TEMPORAL_MACRO_F1 = 0.3027
TEMPORAL_URGENT_BASE_RATE = 0.310
TEMPORAL_DEFAULT_RECALL = 0.031
TEMPORAL_THRESHOLD = 0.0658
TEMPORAL_PRECISION = 0.414


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🐞 Bug Prioritization")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔮 Predict Bug",
        "📂 Batch Prediction",
        "🔍 Model Insights"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
### Project

**Intelligent Bug Prioritization System**

Machine-learning based bug triage using:

- TF-IDF
- Categorical features
- XGBoost
- Threshold optimization
- Cost-sensitive decisions
- Human-in-the-loop review
"""
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🐞 Intelligent Bug Prioritization System")

    st.markdown(
        """
        ## AI-Assisted Bug Triage

        This project explores how machine learning can assist
        software-engineering teams in prioritizing bug reports.

        The system was designed to:

        - Predict bug priority from **P1–P5**
        - Identify potentially **urgent bugs**
        - Optimize the urgent-review probability threshold
        - Analyze precision–recall trade-offs
        - Incorporate cost-sensitive decision making
        - Support **human-in-the-loop** triage
        """
    )

    st.divider()

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
            "Urgent Recall @ 0.097",
            "80.07%"
        )

    st.divider()

    st.subheader("🔄 System Workflow")

    st.markdown(
        """
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
        """
    )

    st.divider()

    st.subheader("🎯 Why Threshold Optimization?")

    st.markdown(
        """
        At the default probability threshold of **0.50**, the
        binary classifier identifies relatively few urgent bugs.

        Because missing an urgent bug can be costly, the decision
        threshold can be lowered to increase urgent-bug recall.

        However, lowering the threshold also sends more normal
        bugs to the human-review queue.

        Therefore, the deployment problem is a
        **precision–recall and operational-capacity trade-off**.
        """
    )

    st.info(
        """
        **Human-in-the-loop:** The model provides a recommendation.
        A human triager makes the final priority decision.
        """
    )

    st.divider()

    st.subheader("📌 Important Deployment Consideration")

    st.warning(
        """
        This GitHub demo presents the project's evaluation results
        and decision framework. The deployed demo does not load
        the trained model artifacts, so predictions on the
        interactive pages are illustrative rather than outputs
        from the trained XGBoost model.
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Model Dashboard")

    st.markdown(
        """
        Overview of the model's evaluation performance and
        different urgent-bug threshold strategies.
        """
    )

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Multiclass Accuracy",
            f"{MULTICLASS_ACCURACY:.2%}"
        )

    with col2:
        st.metric(
            "Multiclass Macro-F1",
            f"{MULTICLASS_MACRO_F1:.3f}"
        )

    with col3:
        st.metric(
            "Binary Accuracy",
            f"{BINARY_ACCURACY:.2%}"
        )

    st.divider()

    st.subheader("🎯 Threshold Strategies")

    threshold_df = pd.DataFrame(
        {
            "Strategy": [
                "Default",
                "80% Recall Target",
                "Cost-Minimizing"
            ],
            "Threshold": [
                DEFAULT_THRESHOLD,
                RECALL_THRESHOLD,
                COST_THRESHOLD
            ],
            "Precision": [
                DEFAULT_PRECISION,
                RECALL_TARGET_PRECISION,
                COST_PRECISION
            ],
            "Recall": [
                DEFAULT_RECALL,
                RECALL_TARGET_RECALL,
                COST_RECALL
            ]
        }
    )

    st.dataframe(
        threshold_df.style.format(
            {
                "Threshold": "{:.4f}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🎯 80% Recall Operating Point")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Threshold",
            f"{RECALL_THRESHOLD:.3f}"
        )

    with col2:
        st.metric(
            "Urgent Recall",
            f"{RECALL_TARGET_RECALL:.2%}"
        )

    with col3:
        st.metric(
            "Urgent Precision",
            f"{RECALL_TARGET_PRECISION:.2%}"
        )

    st.markdown(
        f"""
        At a threshold of **{RECALL_THRESHOLD:.3f}**, the evaluation
        data produced approximately **80% urgent recall**.

        This is one operating point on the precision–recall curve.
        The appropriate threshold depends on review capacity and
        the relative costs of false negatives and false positives.
        """
    )

    st.divider()

    st.subheader("👤 Human-in-the-Loop")

    st.markdown(
        """
        **Model prediction → Urgent screening → Human review → Final decision**

        The model is designed as a decision-support system.

        It does not autonomously determine the final bug priority.
        """
    )

    st.warning(
        """
        False negatives are particularly important because an
        urgent bug incorrectly classified as normal may be delayed.
        """
    )


# ============================================================
# SINGLE BUG DEMO
# ============================================================

elif page == "🔮 Predict Bug":

    st.title("🔮 Single Bug Prediction Demo")

    st.markdown(
        """
        Enter bug information to explore how an urgent-screening
        decision could be made.

        **Note:** This page is an illustrative demo and does not
        load the trained XGBoost model.
        """
    )

    st.warning(
        """
        Demo mode: the probability shown below is generated using
        a transparent illustrative scoring function. It should not
        be interpreted as an actual prediction from the trained model.
        """
    )

    st.divider()

    st.subheader("📝 Bug Summary")

    summary = st.text_area(
        "Describe the bug",
        placeholder=(
            "Example: Browser crashes when opening multiple tabs..."
        ),
        height=150
    )

    st.subheader("🏷️ Bug Attributes")

    col1, col2 = st.columns(2)

    with col1:

        bug_type = st.selectbox(
            "Type",
            [
                "defect",
                "enhancement",
                "task",
                "blocker"
            ]
        )

        component = st.selectbox(
            "Component",
            [
                "Browser",
                "Core",
                "UI",
                "Network",
                "Database"
            ]
        )

    with col2:

        operating_system = st.selectbox(
            "Operating System",
            [
                "Windows",
                "Linux",
                "Mac OS",
                "Android",
                "Other"
            ]
        )

        version = st.selectbox(
            "Version",
            [
                "Latest",
                "Current",
                "Older",
                "Unknown"
            ]
        )

    st.divider()

    st.subheader("🎚️ Urgent Routing Threshold")

    selected_threshold = st.slider(
        "Urgent probability threshold",
        min_value=0.01,
        max_value=0.50,
        value=RECALL_THRESHOLD,
        step=0.01
    )

    st.caption(
        f"Current threshold: **{selected_threshold:.2f}**"
    )

    st.markdown(
        """
        Lower thresholds increase recall but also increase the
        number of bugs sent for human review.
        """
    )

    predict_button = st.button(
        "🚀 Run Demo Prediction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        if not summary.strip():

            st.warning(
                "Please enter a bug summary."
            )

        else:

            # ------------------------------------------------
            # ILLUSTRATIVE SCORING FUNCTION
            # ------------------------------------------------

            text = summary.lower()

            urgent_keywords = [
                "crash",
                "crashes",
                "data loss",
                "security",
                "cannot start",
                "failure",
                "broken",
                "freeze",
                "freezes",
                "hang",
                "critical",
                "blocker"
            ]

            keyword_score = sum(
                word in text
                for word in urgent_keywords
            )

            length_score = min(
                len(text) / 500,
                1
            )

            probability = (
                0.05
                + 0.08 * keyword_score
                + 0.05 * length_score
            )

            if bug_type == "blocker":
                probability += 0.10

            if component == "Core":
                probability += 0.03

            probability = float(
                np.clip(
                    probability,
                    0.01,
                    0.95
                )
            )

            # ------------------------------------------------
            # ILLUSTRATIVE PRIORITY
            # ------------------------------------------------

            if probability >= 0.70:
                priority = "P1"

            elif probability >= 0.50:
                priority = "P2"

            elif probability >= 0.30:
                priority = "P3"

            elif probability >= 0.15:
                priority = "P4"

            else:
                priority = "P5"

            urgent = probability >= selected_threshold

            confidence = min(
                0.55 + abs(probability - 0.50),
                0.95
            )

            st.divider()

            st.subheader("📊 Demo Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Illustrative Priority",
                    priority
                )

            with col2:
                st.metric(
                    "Illustrative Confidence",
                    f"{confidence:.1%}"
                )

            with col3:
                st.metric(
                    "Illustrative Urgent Probability",
                    f"{probability:.1%}"
                )

            if urgent:

                st.error(
                    "🚨 URGENT REVIEW"
                )

                st.info(
                    """
                    The illustrative probability exceeds the selected
                    threshold, so the bug would be routed to human review.
                    """
                )

            else:

                st.success(
                    "✅ NORMAL"
                )

                st.info(
                    """
                    The illustrative probability does not exceed the
                    selected urgent-routing threshold.
                    """
                )

            st.subheader("🧑‍💻 Human Review Guidance")

            st.markdown(
                f"""
                **Illustrative model recommendation**

                - Priority: **{priority}**
                - Urgent probability: **{probability:.1%}**
                - Threshold: **{selected_threshold:.2f}**
                - Route: **{"URGENT REVIEW" if urgent else "NORMAL"}**

                The final triage decision should remain with a human
                engineering reviewer.
                """
            )


# ============================================================
# BATCH PREDICTION DEMO
# ============================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Bug Prediction Demo")

    st.markdown(
        """
        Upload a CSV containing bug reports to explore an
        illustrative urgent-routing workflow.

        **Required columns:**

        `Summary`, `Type`, `Component`, `OS`, `Version`
        """
    )

    st.warning(
        """
        Demo mode: uploaded bugs are processed using an illustrative
        scoring function. The trained XGBoost model is not loaded.
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload a CSV file to start."
        )

    else:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Successfully loaded {len(df):,} bug reports."
        )

        required_columns = [
            "Summary",
            "Type",
            "Component",
            "OS",
            "Version"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            with st.expander("👀 Preview uploaded data"):

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

            st.divider()

            st.subheader("🎚️ Urgent Routing Threshold")

            selected_threshold = st.slider(
                "Choose probability threshold",
                min_value=0.01,
                max_value=0.50,
                value=RECALL_THRESHOLD,
                step=0.01
            )

            run_prediction = st.button(
                "🚀 Run Demo Batch Prediction",
                type="primary",
                use_container_width=True
            )

            if run_prediction:

                results = df.copy()

                probabilities = []

                priorities = []

                for _, row in df.iterrows():

                    text = str(
                        row["Summary"]
                    ).lower()

                    urgent_keywords = [
                        "crash",
                        "crashes",
                        "data loss",
                        "security",
                        "cannot start",
                        "failure",
                        "broken",
                        "freeze",
                        "hang",
                        "critical",
                        "blocker"
                    ]

                    keyword_score = sum(
                        word in text
                        for word in urgent_keywords
                    )

                    probability = (
                        0.05
                        + 0.08 * keyword_score
                        + 0.05 * min(
                            len(text) / 500,
                            1
                        )
                    )

                    if str(row["Type"]).lower() == "blocker":
                        probability += 0.10

                    probability = float(
                        np.clip(
                            probability,
                            0.01,
                            0.95
                        )
                    )

                    probabilities.append(
                        probability
                    )

                    if probability >= 0.70:
                        priorities.append("P1")

                    elif probability >= 0.50:
                        priorities.append("P2")

                    elif probability >= 0.30:
                        priorities.append("P3")

                    elif probability >= 0.15:
                        priorities.append("P4")

                    else:
                        priorities.append("P5")

                results["Predicted_Priority_Demo"] = priorities

                results["Urgent_Probability_Demo"] = (
                    probabilities
                )

                results["Urgent_Review"] = np.where(
                    results["Urgent_Probability_Demo"]
                    >= selected_threshold,
                    "URGENT",
                    "NORMAL"
                )

                results["Threshold_Used"] = (
                    selected_threshold
                )

                st.divider()

                st.subheader("📊 Batch Results")

                total_bugs = len(results)

                urgent_count = (
                    results["Urgent_Review"]
                    .eq("URGENT")
                    .sum()
                )

                routing_rate = (
                    urgent_count / total_bugs
                    if total_bugs > 0
                    else 0
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Bugs",
                        f"{total_bugs:,}"
                    )

                with col2:
                    st.metric(
                        "Urgent",
                        f"{urgent_count:,}"
                    )

                with col3:
                    st.metric(
                        "Routing Rate",
                        f"{routing_rate:.2%}"
                    )

                with col4:
                    st.metric(
                        "Threshold",
                        f"{selected_threshold:.2f}"
                    )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Demo Predictions",
                    data=csv_data,
                    file_name="bug_predictions_demo.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "🔍 Model Insights":

    st.title("🔍 Model Insights")

    st.markdown(
        """
        Explore model evaluation, threshold selection,
        cost-sensitive decisions and temporal robustness.
        """
    )

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Multiclass Accuracy",
            f"{MULTICLASS_ACCURACY:.2%}"
        )

    with col2:
        st.metric(
            "Multiclass Macro-F1",
            f"{MULTICLASS_MACRO_F1:.3f}"
        )

    with col3:
        st.metric(
            "Binary Accuracy",
            f"{BINARY_ACCURACY:.2%}"
        )

    st.divider()

    st.subheader("🚨 Default Urgent Classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Urgent Precision",
            f"{DEFAULT_PRECISION:.2%}"
        )

    with col2:
        st.metric(
            "Urgent Recall",
            f"{DEFAULT_RECALL:.2%}"
        )

    with col3:
        st.metric(
            "Urgent F1",
            f"{DEFAULT_F1:.3f}"
        )

    st.markdown(
        """
        At the default 0.50 threshold, overall binary accuracy
        is relatively high, but urgent-bug recall is low.

        This illustrates why accuracy alone is not sufficient for
        an imbalanced urgent-bug screening problem.
        """
    )

    st.divider()

    st.subheader("🎯 Threshold Strategies")

    threshold_df = pd.DataFrame(
        {
            "Strategy": [
                "Default",
                "80% Recall Target",
                "Cost-Minimizing"
            ],
            "Threshold": [
                DEFAULT_THRESHOLD,
                RECALL_THRESHOLD,
                COST_THRESHOLD
            ],
            "Precision": [
                DEFAULT_PRECISION,
                RECALL_TARGET_PRECISION,
                COST_PRECISION
            ],
            "Recall": [
                DEFAULT_RECALL,
                RECALL_TARGET_RECALL,
                COST_RECALL
            ]
        }
    )

    st.dataframe(
        threshold_df.style.format(
            {
                "Threshold": "{:.3f}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📊 Precision–Recall Trade-Off")

    plot_df = pd.DataFrame(
        {
            "Threshold": [
                0.50,
                0.30,
                0.20,
                0.10,
                0.097,
                0.08
            ],
            "Precision": [
                0.571,
                0.454,
                0.375,
                0.247,
                0.2435,
                0.2171
            ],
            "Recall": [
                0.1237,
                0.375,
                0.471,
                0.780,
                0.8007,
                0.8729
            ]
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        plot_df["Threshold"],
        plot_df["Precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        plot_df["Threshold"],
        plot_df["Recall"],
        marker="o",
        label="Recall"
    )

    ax.set_xlabel(
        "Probability Threshold"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Precision vs Recall"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.subheader("🎯 80% Recall Operating Point")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Threshold",
            f"{RECALL_THRESHOLD:.3f}"
        )

    with col2:
        st.metric(
            "Urgent Recall",
            f"{RECALL_TARGET_RECALL:.2%}"
        )

    with col3:
        st.metric(
            "Urgent Precision",
            f"{RECALL_TARGET_PRECISION:.2%}"
        )

    st.info(
        f"""
        A threshold of **{RECALL_THRESHOLD:.3f}** corresponds to
        approximately **80% urgent recall** on the evaluation data.

        This is an operating point rather than a universally optimal
        threshold. It should be adjusted based on review capacity
        and business costs.
        """
    )

    st.divider()

    st.subheader("💰 Cost-Sensitive Decision Making")

    st.markdown(
        f"""
        The project used a cost framework where:

        **False Negative cost = {FN_COST}**

        **False Positive cost = {FP_COST}**

        Therefore, the assumed cost of missing an urgent bug is
        **{FN_COST / FP_COST:.0f}×** the cost of an unnecessary
        urgent review.
        """
    )

    st.warning(
        """
        The 10:1 cost ratio is a modeling assumption. In production,
        these costs should be estimated using actual engineering,
        incident and triage costs.
        """
    )

    st.divider()

    st.subheader("🕒 Temporal Robustness & Distribution Shift")

    temporal_comparison = pd.DataFrame(
        {
            "Evaluation": [
                "Random Split",
                "Temporal Split"
            ],
            "Accuracy": [
                MULTICLASS_ACCURACY,
                TEMPORAL_ACCURACY
            ],
            "Macro F1": [
                MULTICLASS_MACRO_F1,
                TEMPORAL_MACRO_F1
            ]
        }
    )

    st.dataframe(
        temporal_comparison.style.format(
            {
                "Accuracy": "{:.2%}",
                "Macro F1": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        f"""
        Performance decreased under temporal distribution shift:

        - Accuracy: **66.75% → 61.70%**
        - Macro-F1: **0.326 → 0.303**

        The urgent-bug base rate increased to approximately
        **31.0%**, while default-threshold urgent recall dropped
        to **3.1%**.

        A recalibrated threshold of **{TEMPORAL_THRESHOLD:.4f}**
        produced approximately **80% recall** at **41.4% precision**
        on the temporal evaluation.
        """
    )

    st.caption(
        """
        Note: the temporal split uses updated_date because a reliable
        creation timestamp was unavailable. Therefore, the future
        window may include reopened or escalated bugs rather than
        exclusively newly created reports.
        """
    )

    st.divider()

    st.subheader("👤 Human-in-the-Loop Deployment")

    st.markdown(
        """
        **Bug Report**

        ↓

        **ML Priority Prediction**

        ↓

        **Urgent Probability**

        ↓

        **Threshold**

        ↓

        **Urgent Review Queue**

        ↓

        **Human Triager**

        ↓

        **Final Priority Decision**
        """
    )

    st.info(
        """
        The model is a decision-support system. It does not
        autonomously determine the final bug priority.
        """
    )

    st.divider()

    st.subheader("🎯 Product Interpretation")

    st.markdown(
        """
        The deployment problem is not simply maximizing accuracy.

        A lower threshold can increase the number of urgent bugs
        identified, but it also increases the workload placed on
        human triagers.

        Therefore, threshold selection should consider:

        - Cost of missed urgent bugs
        - Cost of unnecessary reviews
        - Human-review capacity
        - Changes in the urgent-bug base rate
        - Distribution shift over time

        In production, the threshold should be monitored and
        recalibrated as the data distribution changes.
        """
    )


