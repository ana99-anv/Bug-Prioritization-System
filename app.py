

# ============================================================
# APP.PY — HOME PAGE
# ============================================================

app_code = r'''
import streamlit as st

st.set_page_config(
    page_title="Bug Prioritization System",
    page_icon="🐞",
    layout="wide"
)

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

col1, col2, col3, col4 = st.columns(4)

col1.metric("Multiclass Accuracy", "66.75%")
col2.metric("Multiclass Macro-F1", "0.326")
col3.metric("Binary Accuracy", "85.90%")
col4.metric("Urgent Recall", "80.07%")

st.divider()

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

st.markdown("""
### 📌 Navigation

Use the sidebar to access:

📊 **Dashboard**

🔮 **Predict Bug**

📂 **Batch Prediction**

🔍 **Model Insights**
""")
'''

with open(
    os.path.join(base_path, "app.py"),
    "w",
    encoding="utf-8"
) as f:
    f.write(app_code)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

dashboard_code = r'''
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Dashboard")

metadata = joblib.load(
    "saved_models/metadata.pkl"
)

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Multiclass Accuracy",
        f"{metadata['multiclass_accuracy']:.2%}"
    )

with col2:
    st.metric(
        "Multiclass Macro-F1",
        f"{metadata['multiclass_macro_f1']:.3f}"
    )

with col3:
    st.metric(
        "Binary Accuracy",
        f"{metadata['binary_accuracy']:.2%}"
    )

st.divider()

st.subheader("🎯 Threshold Strategies")

threshold_df = pd.DataFrame({

    "Strategy": [
        "Default",
        "80% Recall Target",
        "Cost-Minimizing"
    ],

    "Threshold": [
        metadata["default_threshold"],
        metadata["urgent_recall_threshold"],
        metadata["cost_minimizing_threshold"]
    ],

    "Precision": [
        metadata["urgent_precision_default"],
        metadata["recall_target_precision"],
        0.2171
    ],

    "Recall": [
        metadata["urgent_recall_default"],
        metadata["recall_target_recall"],
        0.8729
    ]
})

st.dataframe(
    threshold_df.style.format({
        "Threshold": "{:.4f}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("🎯 80% Recall Operating Point")

st.metric(
    "Threshold",
    f"{metadata['urgent_recall_threshold']:.3f}"
)

st.markdown(
    f"""
    At this operating point:

    - Urgent precision: **{metadata['recall_target_precision']:.2%}**
    - Urgent recall: **{metadata['recall_target_recall']:.2%}**

    This threshold represents one operating point on the
    precision-recall trade-off and can be adjusted according
    to review capacity and business costs.
    """
)

st.divider()

st.subheader("👤 Human-in-the-Loop")

st.markdown("""
**Model prediction → Urgent screening → Human review → Final decision**

The model is designed as a decision-support system.
The final triage decision remains with a human reviewer.
""")

st.warning("""
False negatives are particularly important because an urgent
bug incorrectly classified as normal may be delayed.
""")
'''

with open(
    os.path.join(pages_path, "1_Dashboard.py"),
    "w",
    encoding="utf-8"
) as f:
    f.write(dashboard_code)


# ============================================================
# PAGE 2 — PREDICT BUG
# ============================================================

predict_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse

st.set_page_config(
    page_title="Predict Bug",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Single Bug Prediction")

st.markdown("""
Enter the bug details below. The model will provide:

- **Predicted Priority (P1–P5)**
- **Priority confidence**
- **Urgent probability**
- **Urgent / Normal routing recommendation**
- **Human-in-the-loop review recommendation**
""")

@st.cache_resource
def load_models():

    model = joblib.load(
        "saved_models/model.pkl"
    )

    binary_model = joblib.load(
        "saved_models/binary_model.pkl"
    )

    tfidf = joblib.load(
        "saved_models/tfidf.pkl"
    )

    encoder = joblib.load(
        "saved_models/encoder.pkl"
    )

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

    label_encoder = joblib.load(
        "saved_models/label_encoder.pkl"
    )

    metadata = joblib.load(
        "saved_models/metadata.pkl"
    )

    return (
        model,
        binary_model,
        tfidf,
        encoder,
        scaler,
        label_encoder,
        metadata
    )


(
    model,
    binary_model,
    tfidf,
    encoder,
    scaler,
    label_encoder,
    metadata
) = load_models()


default_threshold = float(
    metadata.get(
        "urgent_recall_threshold",
        0.097
    )
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


def clean_options(categories):

    return [
        str(value)
        for value in categories
        if str(value).lower()
        not in ["missing", "nan", "none"]
    ]


type_options = clean_options(
    encoder.categories_[0]
)

component_options = clean_options(
    encoder.categories_[1]
)

os_options = clean_options(
    encoder.categories_[2]
)

version_options = clean_options(
    encoder.categories_[3]
)


col1, col2 = st.columns(2)

with col1:

    bug_type = st.selectbox(
        "Type",
        options=type_options
    )

    component = st.selectbox(
        "Component",
        options=component_options
    )

with col2:

    operating_system = st.selectbox(
        "OS",
        options=os_options
    )

    version = st.selectbox(
        "Version",
        options=version_options
    )


st.divider()

st.subheader("🎚️ Urgent Routing Threshold")

selected_threshold = st.slider(
    "Urgent probability threshold",
    min_value=0.01,
    max_value=0.50,
    value=float(
        np.clip(
            default_threshold,
            0.01,
            0.50
        )
    ),
    step=0.01
)

st.caption(
    f"Current threshold: **{selected_threshold:.2f}**"
)

st.markdown("""
A lower threshold prioritizes **recall** and catches more
potentially urgent bugs, while a higher threshold prioritizes
**precision** and reduces the human-review workload.
""")

st.divider()

predict_button = st.button(
    "🚀 Predict Bug Priority",
    type="primary",
    use_container_width=True
)


if predict_button:

    if not summary.strip():

        st.warning(
            "Please enter a bug summary before running prediction."
        )

        st.stop()

    with st.spinner("Analyzing bug..."):

        clean_summary = (
            summary
            .lower()
            .strip()
        )

        X_text = tfidf.transform(
            [clean_summary]
        )

        X_cat_df = pd.DataFrame({
            "Type": [bug_type],
            "Component": [component],
            "OS": [operating_system],
            "Version": [version]
        })

        X_cat = encoder.transform(
            X_cat_df
        )

        X_num_df = pd.DataFrame({

            "summary_length": [
                len(clean_summary)
            ],

            "word_count": [
                len(clean_summary.split())
            ]
        })

        X_num = scaler.transform(
            X_num_df
        )

        X_num = sparse.csr_matrix(
            X_num
        )

        X_final = sparse.hstack([
            X_text,
            X_cat,
            X_num
        ]).tocsr()

        multiclass_pred = model.predict(
            X_final
        )

        predicted_priority = (
            label_encoder.inverse_transform(
                multiclass_pred
            )[0]
        )

        multiclass_proba = (
            model.predict_proba(
                X_final
            )[0]
        )

        priority_confidence = (
            multiclass_proba.max()
        )

        binary_proba = float(
            binary_model
            .predict_proba(X_final)[0, 1]
        )

        urgent_prediction = int(
            binary_proba >= selected_threshold
        )

        if urgent_prediction == 1:

            routing_decision = "URGENT REVIEW"

            routing_message = (
                "This bug should be routed to a human "
                "triager for urgent review."
            )

        else:

            routing_decision = "NORMAL"

            routing_message = (
                "This bug does not meet the current urgent "
                "routing threshold."
            )

    st.divider()

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Priority",
            predicted_priority
        )

    with col2:
        st.metric(
            "Priority Confidence",
            f"{priority_confidence:.1%}"
        )

    with col3:
        st.metric(
            "Urgent Probability",
            f"{binary_proba:.1%}"
        )

    if urgent_prediction == 1:

        st.error(
            f"🚨 **{routing_decision}**"
        )

    else:

        st.success(
            f"✅ **{routing_decision}**"
        )

    st.info(
        f"""
        **Urgent probability:** {binary_proba:.2%}

        **Threshold:** {selected_threshold:.2f}

        {routing_message}

        **Human-in-the-loop:** The model provides a
        recommendation; the final triage decision remains
        with a human reviewer.
        """
    )

    st.subheader("📈 Priority Probabilities")

    priority_classes = (
        label_encoder.classes_
    )

    probability_df = pd.DataFrame({
        "Priority": priority_classes,
        "Probability": multiclass_proba
    })

    probability_df["Probability"] = (
        probability_df["Probability"]
        .round(4)
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🧑‍💻 Human Review Guidance")

    st.markdown(f"""
    **Model recommendation**

    - Predicted priority: **{predicted_priority}**
    - Priority confidence: **{priority_confidence:.1%}**
    - Urgent probability: **{binary_proba:.1%}**
    - Routing threshold: **{selected_threshold:.2f}**
    - Recommended route: **{routing_decision}**

    The prediction is intended to **assist**, not replace,
    the engineering triager.
    """)
'''

with open(
    os.path.join(pages_path, "2_Predict_Bug.py"),
    "w",
    encoding="utf-8"
) as f:
    f.write(predict_code)


# ============================================================
# PAGE 3 — BATCH PREDICTION
# ============================================================

batch_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Bug Prediction")

st.markdown("""
Upload a CSV containing bug reports and the model will predict:

- **Bug Priority (P1–P5)**
- **Urgent probability**
- **Urgent / Normal routing**
- **Prediction confidence**
""")

@st.cache_resource
def load_models():

    model = joblib.load(
        "saved_models/model.pkl"
    )

    binary_model = joblib.load(
        "saved_models/binary_model.pkl"
    )

    tfidf = joblib.load(
        "saved_models/tfidf.pkl"
    )

    encoder = joblib.load(
        "saved_models/encoder.pkl"
    )

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

    label_encoder = joblib.load(
        "saved_models/label_encoder.pkl"
    )

    metadata = joblib.load(
        "saved_models/metadata.pkl"
    )

    return (
        model,
        binary_model,
        tfidf,
        encoder,
        scaler,
        label_encoder,
        metadata
    )


(
    model,
    binary_model,
    tfidf,
    encoder,
    scaler,
    label_encoder,
    metadata
) = load_models()


default_threshold = float(
    metadata.get(
        "urgent_recall_threshold",
        0.097
    )
)

st.divider()

st.subheader("📄 Upload Bug CSV")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="batch_csv"
)

if uploaded_file is None:

    st.info(
        "👆 Please upload a CSV file to start batch prediction."
    )

    st.stop()


df = pd.read_csv(uploaded_file)

st.success(
    f"Successfully loaded {len(df):,} bug reports."
)

with st.expander("👀 Preview uploaded data"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


required_columns = [
    "Summary",
    "Type",
    "Component",
    "OS",
    "Version"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


st.divider()

st.subheader("🎚️ Urgent Routing Threshold")

selected_threshold = st.slider(
    "Choose probability threshold",
    min_value=0.01,
    max_value=0.50,
    value=float(
        np.clip(
            default_threshold,
            0.01,
            0.50
        )
    ),
    step=0.01
)

st.caption(
    f"Current threshold: **{selected_threshold:.2f}**"
)


run_prediction = st.button(
    "🚀 Run Batch Prediction",
    type="primary",
    use_container_width=True
)


if run_prediction:

    with st.spinner("Running predictions..."):

        clean_summary = (
            df["Summary"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        X_text = tfidf.transform(
            clean_summary
        )

        X_cat_df = pd.DataFrame({

            "Type": (
                df["Type"]
                .fillna("missing")
                .astype(str)
            ),

            "Component": (
                df["Component"]
                .fillna("missing")
                .astype(str)
            ),

            "OS": (
                df["OS"]
                .fillna("missing")
                .astype(str)
            ),

            "Version": (
                df["Version"]
                .fillna("missing")
                .astype(str)
            )
        })

        X_cat = encoder.transform(
            X_cat_df
        )

        X_num_df = pd.DataFrame({

            "summary_length": (
                clean_summary.str.len()
            ),

            "word_count": (
                clean_summary
                .str.split()
                .str.len()
            )
        })

        X_num = scaler.transform(
            X_num_df
        )

        X_num = sparse.csr_matrix(
            X_num
        )

        X_final = sparse.hstack([
            X_text,
            X_cat,
            X_num
        ]).tocsr()

        multiclass_pred = model.predict(
            X_final
        )

        predicted_priority = (
            label_encoder.inverse_transform(
                multiclass_pred
            )
        )

        multiclass_proba = (
            model.predict_proba(
                X_final
            )
        )

        priority_confidence = (
            multiclass_proba.max(
                axis=1
            )
        )

        binary_proba = (
            binary_model
            .predict_proba(X_final)[:, 1]
        )

        urgent_prediction = (
            binary_proba >= selected_threshold
        ).astype(int)

        results = df.copy()

        results["Predicted_Priority"] = (
            predicted_priority
        )

        results["Priority_Confidence"] = (
            priority_confidence
        )

        results["Urgent_Probability"] = (
            binary_proba
        )

        results["Urgent_Review"] = np.where(
            urgent_prediction == 1,
            "URGENT",
            "NORMAL"
        )

        st.session_state["batch_results"] = results

        st.session_state[
            "batch_threshold"
        ] = selected_threshold


if "batch_results" in st.session_state:

    results = st.session_state[
        "batch_results"
    ]

    threshold_used = st.session_state.get(
        "batch_threshold",
        selected_threshold
    )

    st.divider()

    st.subheader("📊 Prediction Results")

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

    col1.metric(
        "Total Bugs",
        f"{total_bugs:,}"
    )

    col2.metric(
        "Urgent Bugs",
        f"{urgent_count:,}"
    )

    col3.metric(
        "Urgent Routing Rate",
        f"{routing_rate:.2%}"
    )

    col4.metric(
        "Threshold Used",
        f"{threshold_used:.2f}"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Predictions",
        data=csv_data,
        file_name="bug_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
'''

with open(
    os.path.join(pages_path, "3_Batch_Prediction.py"),
    "w",
    encoding="utf-8"
) as f:
    f.write(batch_code)


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================

insights_code = r'''
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Insights",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Model Insights")

st.markdown("""
Explore model performance, threshold selection, cost analysis,
temporal robustness and the human-in-the-loop deployment strategy.
""")

metadata = joblib.load(
    "saved_models/metadata.pkl"
)

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Multiclass Accuracy",
    f"{metadata['multiclass_accuracy']:.2%}"
)

col2.metric(
    "Multiclass Macro-F1",
    f"{metadata['multiclass_macro_f1']:.3f}"
)

col3.metric(
    "Binary Accuracy",
    f"{metadata['binary_accuracy']:.2%}"
)

st.divider()

st.subheader("🚨 Default Urgent Classification")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Urgent Precision",
    f"{metadata['urgent_precision_default']:.2%}"
)

col2.metric(
    "Urgent Recall",
    f"{metadata['urgent_recall_default']:.2%}"
)

col3.metric(
    "Urgent F1",
    f"{metadata['urgent_f1_default']:.3f}"
)

st.markdown("""
At the default 0.50 probability threshold, the binary model
has high overall accuracy but catches relatively few urgent
bugs. This motivates threshold tuning.
""")

st.divider()

st.subheader("🎯 Threshold Strategies")

threshold_df = pd.DataFrame({

    "Strategy": [
        "Default",
        "80% Recall Target",
        "Cost-Minimizing"
    ],

    "Threshold": [
        metadata["default_threshold"],
        metadata["urgent_recall_threshold"],
        metadata["cost_minimizing_threshold"]
    ],

    "Precision": [
        metadata["urgent_precision_default"],
        metadata["recall_target_precision"],
        0.2171
    ],

    "Recall": [
        metadata["urgent_recall_default"],
        metadata["recall_target_recall"],
        0.8729
    ]
})

display_df = threshold_df.copy()

display_df["Threshold"] = (
    display_df["Threshold"]
    .map(lambda x: f"{x:.3f}")
)

display_df["Precision"] = (
    display_df["Precision"]
    .map(lambda x: f"{x:.2%}")
)

display_df["Recall"] = (
    display_df["Recall"]
    .map(lambda x: f"{x:.2%}")
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("📊 Precision–Recall Trade-Off")

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
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
    "Precision vs Recall at Selected Operating Points"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(fig)

plt.close(fig)

st.divider()

st.subheader("🎯 80% Recall Operating Point")

recommended_threshold = (
    metadata["urgent_recall_threshold"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Threshold",
    f"{recommended_threshold:.3f}"
)

col2.metric(
    "Urgent Recall",
    f"{metadata['recall_target_recall']:.2%}"
)

col3.metric(
    "Urgent Precision",
    f"{metadata['recall_target_precision']:.2%}"
)

st.info(
    f"""
    The threshold of **{recommended_threshold:.3f}**
    corresponds to approximately **80% urgent recall**
    on the evaluation data.

    This operating point can be adjusted according to
    review capacity and the relative cost of false negatives
    and false positives.
    """
)

st.divider()

st.subheader("💰 Cost-Sensitive Decision Making")

fn_cost = metadata["fn_cost"]
fp_cost = metadata["fp_cost"]

st.markdown(
    f"""
    The cost framework assumes:

    **False Negative cost = {fn_cost}**

    **False Positive cost = {fp_cost}**

    Therefore, a missed urgent bug is treated as
    **{fn_cost / fp_cost:.0f}× more costly** than an unnecessary
    urgent review.
    """
)

cost_comparison = pd.DataFrame({

    "Strategy": [
        "Default",
        "80% Recall Target",
        "Cost-Minimizing"
    ],

    "Threshold": [
        metadata["default_threshold"],
        metadata["urgent_recall_threshold"],
        metadata["cost_minimizing_threshold"]
    ],

    "Urgent Recall": [
        metadata["urgent_recall_default"],
        metadata["recall_target_recall"],
        0.8729
    ],

    "Precision": [
        metadata["urgent_precision_default"],
        metadata["recall_target_precision"],
        0.2171
    ]
})

st.dataframe(
    cost_comparison.style.format({
        "Threshold": "{:.3f}",
        "Urgent Recall": "{:.2%}",
        "Precision": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.warning("""
The 10:1 cost ratio is a modeling assumption. Before
production deployment, these costs should be calibrated using
actual engineering and triage costs.
""")

st.divider()

st.subheader("👤 Human-in-the-Loop")

st.markdown("""
### Deployment workflow

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
""")

st.info("""
The model is a decision-support system. It does not
autonomously determine the final bug priority.
""")

st.divider()

st.subheader("🎯 Product Interpretation")

st.markdown("""
The key deployment trade-off is not simply accuracy.

A lower threshold increases the number of potentially urgent
bugs caught by the system, but also increases the number of
normal bugs sent to human reviewers.

The threshold can therefore be adjusted according to
triage capacity and the relative cost of false negatives
versus false positives.
""")

st.divider()

st.subheader("🕒 Temporal Robustness & Distribution Shift")

st.markdown("""
A temporal holdout was evaluated to test whether model
performance remains stable when predicting on data from a
later time period.

This is important because bug characteristics, components,
versions, and priority patterns can change over time.
""")

temporal_comparison = pd.DataFrame({

    "Evaluation": [
        "Random Split",
        "Temporal Split"
    ],

    "Accuracy": [
        0.6675,
        0.6170
    ],

    "Macro F1": [
        0.3256,
        0.3027
    ]
})

st.dataframe(
    temporal_comparison.style.format({
        "Accuracy": "{:.2%}",
        "Macro F1": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.warning("""
**Key takeaway:** Performance decreased under temporal
distribution shift, with accuracy falling from **66.75% to
61.70%** and Macro-F1 from **0.326 to 0.303**.

Urgent-bug base rate rose from **10.4% to 31.0%**, and the
default threshold's urgent recall dropped to **3.1%**.

A recalibrated threshold of **0.0658** restored approximately
**80% recall** at **41.4% precision**, showing that threshold
calibration may need to be revisited as the data distribution
changes.
""")

st.caption("""
Note: the temporal split uses updated_date because a creation
timestamp was unavailable. The future window may therefore
include reopened or escalated bugs rather than purely new
reports.
""")
'''



