from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Online Shoppers | Purchase Intent System",
    page_icon="shopping-cart",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "v1"
RAW_DATA_PATH = ROOT / "online_shoppers_intention.csv"

MODEL_INPUT_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "TotalSessionDuration",
    "ProductInfoRatio",
    "EngagementScore",
    "Month_sin",
    "Month_cos",
]

CATEGORICAL_COLUMNS = ["VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
MONTH_OPTIONS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BOOL_MAP = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}
MONTH_TO_NUM = {month: index + 1 for index, month in enumerate(MONTH_OPTIONS)}
SERVING_RAW_INPUT_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Month",
]

CALM_BG = "#f7f5ef"
INK = "#0f172a"
TEAL = "#0f766e"
AMBER = "#c28b2c"
SOFT_BLUE = "#dbeafe"
SOFT_GREEN = "#d1fae5"
SOFT_AMBER = "#fef3c7"
SOFT_SLATE = "#e2e8f0"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {CALM_BG} 0%, #ffffff 35%, #f8fafc 100%);
            color: {INK};
        }}
        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 2.8rem;
            max-width: 1450px;
        }}
        h1, h2, h3, h4, h5 {{
            color: {INK};
            letter-spacing: -0.02em;
        }}
        .hero {{
            padding: 2.1rem 2rem 1.7rem 2rem;
            border-radius: 28px;
            background: radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 35%),
                        linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,255,255,0.74));
            border: 1px solid rgba(15,118,110,0.12);
            box-shadow: 0 24px 60px rgba(15,23,42,0.08);
            margin-bottom: 1.5rem;
        }}
        .hero h1 {{
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }}
        .subtle {{
            color: #475569;
            font-size: 1.04rem;
            line-height: 1.7;
        }}
        .pill {{
            display: inline-block;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            background: rgba(15,118,110,0.10);
            color: {TEAL};
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
        }}
        .card {{
            border-radius: 22px;
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148,163,184,0.18);
            padding: 1.05rem 1.05rem 1rem 1.05rem;
            box-shadow: 0 14px 30px rgba(15,23,42,0.05);
        }}
        .metric-label {{
            color: #64748b;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {INK};
            line-height: 1.1;
        }}
        .metric-note {{
            color: #475569;
            margin-top: 0.35rem;
            font-size: 0.94rem;
        }}
        .mini-card {{
            border-radius: 18px;
            background: linear-gradient(180deg, white, #fbfdff);
            border: 1px solid rgba(148,163,184,0.18);
            padding: 1rem;
            height: 100%;
        }}
        .mini-card h4 {{
            margin-bottom: 0.25rem;
        }}
        .step-num {{
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            margin-bottom: 0.7rem;
            background: rgba(15,118,110,0.12);
            color: {TEAL};
        }}
        .step-title {{
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: {INK};
        }}
        .step-text {{
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.45;
        }}
        .section-tag {{
            display: inline-block;
            margin-bottom: 0.7rem;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(15,118,110,0.08);
            color: {TEAL};
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .kpi-box {{
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: white;
            border: 1px solid rgba(148,163,184,0.18);
        }}
        .big-number {{
            font-size: 2.4rem;
            font-weight: 900;
            color: {INK};
            line-height: 1.1;
        }}
        .small-muted {{
            color: #64748b;
            font-size: 0.9rem;
        }}
        .sidebar-note {{
            border-radius: 16px;
            padding: 0.9rem;
            background: rgba(15,118,110,0.07);
            border: 1px solid rgba(15,118,110,0.12);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_drift_report() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "drift_report.csv")


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "X_processed.csv")


@st.cache_data(show_spinner=False)
def load_target() -> pd.Series:
    return pd.read_csv(ARTIFACT_DIR / "y.csv").iloc[:, 0]


@st.cache_data(show_spinner=False)
def clean_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in ["Weekend"]:
        df[col] = df[col].map(BOOL_MAP)

    df["Month_num"] = df["Month"].map(MONTH_TO_NUM)
    if df["Month_num"].isna().any():
        df["Month_num"] = df["Month_num"].fillna(df["Month_num"].mode().iloc[0])

    df["TotalSessionDuration"] = (
        df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    )
    df["ProductInfoRatio"] = df["ProductRelated"] / (df["Informational"] + 1)
    df["EngagementScore"] = df["PageValues"] / (df["TotalSessionDuration"] + 1)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12.0)
    df = df.drop(columns=["Month_num", "Month"], errors="ignore")
    return df[MODEL_INPUT_COLUMNS].copy()


@st.cache_data(show_spinner=False)
def build_preprocessor(sample_frame: pd.DataFrame):
    numeric_columns = [col for col in sample_frame.columns if col not in CATEGORICAL_COLUMNS]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    preprocessor.fit(sample_frame)
    return preprocessor


@st.cache_resource(show_spinner=False)
def train_demo_model(processed_df: pd.DataFrame, target: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        processed_df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }
    coef = pd.Series(model.coef_[0], index=processed_df.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    metrics["top_coefficients"] = coef.head(10)
    metrics["bottom_coefficients"] = coef.tail(10)
    return model, metrics, (X_test, y_test)


@st.cache_data(show_spinner=False)
def target_distribution(target: pd.Series) -> pd.DataFrame:
    counts = target.value_counts().sort_index()
    return pd.DataFrame({"label": ["No Purchase", "Purchase"], "count": [int(counts.get(0, 0)), int(counts.get(1, 0))]})


@st.cache_data(show_spinner=False)
def month_distribution(raw_df: pd.DataFrame) -> pd.DataFrame:
    counts = raw_df["Month"].value_counts().reindex(MONTH_OPTIONS, fill_value=0)
    return pd.DataFrame({"Month": counts.index, "Count": counts.values})


@st.cache_data(show_spinner=False)
def numeric_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]
    summary = raw_df[numeric_cols].describe().T[["mean", "50%", "std"]].rename(columns={"50%": "median"})
    summary = summary.reset_index().rename(columns={"index": "feature"})
    return summary


def make_title(text: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">Versioned ML System</div>
            <div class="pill">Purchase Intent Prediction</div>
            <div class="pill">Recall + F1 Focus</div>
            <h1>{text}</h1>
            <p class="subtle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav(metadata: dict, metrics: dict) -> str:
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Open a section",
        ["Home", "Data Lab", "Prediction Studio", "Serving Hub", "Monitoring Center", "Deploy"],
        index=0,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <strong>Current version</strong><br>
            {metadata['data_version']}<br><br>
            <strong>Rows</strong><br>
            {metadata['total_rows']:,}<br><br>
            <strong>High drift features</strong><br>
            {metadata['features_with_high_drift']}<br><br>
            <strong>Demo recall</strong><br>
            {metrics['recall']:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-note" style="margin-top:0.75rem;">
            <strong>What this app does</strong><br>
            It is a full website-style interface for exploring the project, testing a purchase-intent model,
            and reviewing deployment steps.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def render_home(metadata: dict, drift_report: pd.DataFrame, raw_df: pd.DataFrame, processed_df: pd.DataFrame, target: pd.Series, metrics: dict) -> None:
    positive_count = int(target.sum())
    negative_count = int(len(target) - positive_count)
    positive_rate = float(metadata["positive_rate"])

    make_title(
        "Online Shoppers Purchase Intent Dashboard",
        "A real website-style dashboard for the full ML system: data ingestion, data prep, model training, prediction, monitoring, and deployment.",
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Rows", f"{metadata['total_rows']:,}", "Sessions in the versioned dataset"),
        ("Features", f"{metadata['total_features_after_encoding']}", "Model-ready columns after encoding"),
        ("Positive Rate", f"{positive_rate:.2f}%", "Users who purchased in the dataset"),
        ("High Drift Features", f"{metadata['features_with_high_drift']}", "Features flagged by PSI drift check"),
    ]
    for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Business Goal")
        st.write(
            "Use ML to predict purchase intent so the business can take targeted actions to increase revenue, improve ROI, and reduce waste."
        )
        st.markdown(
            """
            <div class="mini-card">
                <h4>Who uses this system?</h4>
                <div class="step-text">
                    Direct clients are e-commerce platforms, marketing teams, and personalization engines.
                    Indirect clients include analytics consultancies and fraud/payment optimization services.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Key Metrics")
        st.markdown(
            """
            <div class='mini-card' style='margin-bottom:0.9rem;'>
                <h4>Recall</h4>
                <div class='step-text'>Catching potential buyers is more critical than falsely predicting a few non-buyers as buyers.</div>
            </div>
            <div class='mini-card' style='margin-bottom:0.9rem;'>
                <h4>F1 Score</h4>
                <div class='step-text'>Balances precision and recall so the model stays useful for business action.</div>
            </div>
            <div class='mini-card'>
                <h4>Latency</h4>
                <div class='step-text'>Fast predictions matter because this is meant for real-time targeting while the visitor is still active.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### End-to-End System Flow")
    flow_steps = [
        ("1", "Raw Data", "CSV / DB source from Kaggle."),
        ("2", "Versioned Ingestion", "Store each revision as v1, v2, ..."),
        ("3", "Data Pipeline", "Cleaning, encoding, feature engineering, drift checks."),
        ("4", "ML Pipeline", "Split, train, tune, and evaluate Recall/F1."),
        ("5", "Serving Pipeline", "Real-time or batch inference with a threshold."),
        ("6", "Monitoring", "Business KPIs, latency, drift, retraining triggers."),
        ("7", "Artifacts", "Save processed features, metadata, and preprocessor objects."),
    ]
    top_row = st.columns(4)
    bottom_row = st.columns(3)
    for col, step in zip(top_row, flow_steps[:4]):
        num, title, text = step
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="min-height:150px;">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    for col, step in zip(bottom_row, flow_steps[4:]):
        num, title, text = step
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="min-height:150px;">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Business Snapshot")
    pie_left, pie_right = st.columns([1, 1])
    with pie_left:
        target_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["No Purchase", "Purchase"],
                    values=[negative_count, positive_count],
                    hole=0.58,
                    marker=dict(colors=["#cbd5e1", TEAL]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        target_fig.update_layout(
            title="Target Distribution",
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK),
        )
        st.plotly_chart(target_fig, use_container_width=True)
    with pie_right:
        month_df = month_distribution(raw_df)
        month_fig = px.bar(
            month_df,
            x="Month",
            y="Count",
            title="Sessions by Month",
            color="Count",
            color_continuous_scale=[SOFT_BLUE, TEAL],
        )
        month_fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK),
            coloraxis_showscale=False,
        )
        st.plotly_chart(month_fig, use_container_width=True)

    st.markdown("### What Was Engineered")
    feature_cards = st.columns(5)
    feature_details = [
        ("TotalSessionDuration", "Sum of all page durations."),
        ("ProductInfoRatio", "Product focus vs informational browsing."),
        ("EngagementScore", "Value relative to time spent."),
        ("Month_sin", "Seasonality encoded as a cycle."),
        ("Month_cos", "Seasonality encoded as a cycle."),
    ]
    for col, (name, desc) in zip(feature_cards, feature_details):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <h4>{name}</h4>
                    <div class="step-text">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Why the Preprocessing Choice Matters")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Numerical Pipeline</h4>
                <div class='step-text'>Median imputation and standardization keep numeric features stable for ML.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Categorical Pipeline</h4>
                <div class='step-text'>One-hot encoding prevents numeric-looking categories from being treated like ordered values.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pc3:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Versioning</h4>
                <div class='step-text'>Versioned artifacts make it easy to reproduce training, inference, and monitoring later.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Artifact Summary")
    art_cols = st.columns(5)
    artifacts = [
        ("preprocessor.joblib", "Reusable transformation pipeline"),
        ("X_processed.csv", "Encoded and scaled features"),
        ("y.csv", "Target labels"),
        ("drift_report.csv", "Feature drift monitoring"),
        ("metadata.json", "Version summary and stats"),
    ]
    for col, (fname, desc) in zip(art_cols, artifacts):
        with col:
            st.markdown(
                f"""
                <div class='mini-card'>
                    <h4>{fname}</h4>
                    <div class='step-text'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Monitoring Snapshot")
    drift_plot = drift_report.sort_values("psi", ascending=True).tail(10)
    drift_fig = px.bar(
        drift_plot,
        x="psi",
        y="feature",
        orientation="h",
        color="drift_flag",
        color_discrete_map={True: AMBER, False: TEAL},
        title="Top Drift Features (PSI)",
    )
    drift_fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        xaxis_title="PSI",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(drift_fig, use_container_width=True)


def build_input_defaults(raw_df: pd.DataFrame) -> dict:
    defaults = {}
    numeric_defaults = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]
    for col in numeric_defaults:
        defaults[col] = float(raw_df[col].median())
    defaults["OperatingSystems"] = int(raw_df["OperatingSystems"].mode().iloc[0])
    defaults["Browser"] = int(raw_df["Browser"].mode().iloc[0])
    defaults["Region"] = int(raw_df["Region"].mode().iloc[0])
    defaults["TrafficType"] = int(raw_df["TrafficType"].mode().iloc[0])
    defaults["VisitorType"] = raw_df["VisitorType"].mode().iloc[0]
    defaults["Weekend"] = int(raw_df["Weekend"].map(BOOL_MAP).fillna(0).mode().iloc[0])
    defaults["Month"] = raw_df["Month"].mode().iloc[0]
    return defaults


def assemble_prediction_frame(values: dict) -> pd.DataFrame:
    frame = pd.DataFrame([values])
    frame["Weekend"] = frame["Weekend"].map(BOOL_MAP).fillna(0).astype(int)
    month_num = MONTH_TO_NUM.get(frame.loc[0, "Month"], 6)
    frame["Month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    frame["Month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
    frame["TotalSessionDuration"] = (
        frame["Administrative_Duration"] + frame["Informational_Duration"] + frame["ProductRelated_Duration"]
    )
    frame["ProductInfoRatio"] = frame["ProductRelated"] / (frame["Informational"] + 1)
    frame["EngagementScore"] = frame["PageValues"] / (frame["TotalSessionDuration"] + 1)
    frame = frame.drop(columns=["Month"], errors="ignore")
    return frame[MODEL_INPUT_COLUMNS].copy()


def prepare_serving_frame_from_raw(input_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SERVING_RAW_INPUT_COLUMNS if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for serving: {missing}")

    frame = input_df[SERVING_RAW_INPUT_COLUMNS].copy()
    frame["Weekend"] = frame["Weekend"].map(BOOL_MAP).fillna(0).astype(int)
    frame["Month_num"] = frame["Month"].map(MONTH_TO_NUM)
    if frame["Month_num"].isna().any():
        frame["Month_num"] = frame["Month_num"].fillna(frame["Month_num"].mode().iloc[0])

    frame["TotalSessionDuration"] = (
        frame["Administrative_Duration"] + frame["Informational_Duration"] + frame["ProductRelated_Duration"]
    )
    frame["ProductInfoRatio"] = frame["ProductRelated"] / (frame["Informational"] + 1)
    frame["EngagementScore"] = frame["PageValues"] / (frame["TotalSessionDuration"] + 1)
    frame["Month_sin"] = np.sin(2 * np.pi * frame["Month_num"] / 12.0)
    frame["Month_cos"] = np.cos(2 * np.pi * frame["Month_num"] / 12.0)
    frame = frame.drop(columns=["Month", "Month_num"], errors="ignore")
    return frame[MODEL_INPUT_COLUMNS].copy()


def render_data_lab(raw_df: pd.DataFrame, processed_df: pd.DataFrame, target: pd.Series) -> None:
    make_title(
        "Data Lab",
        "Explore the raw sessions and processed features. This page is for checking how the dataset behaves before and after preprocessing.",
    )
    left, right = st.columns([1.1, 0.9])
    with left:
        feature = st.selectbox(
            "Choose a raw numeric feature",
            [
                "Administrative",
                "Administrative_Duration",
                "Informational",
                "Informational_Duration",
                "ProductRelated",
                "ProductRelated_Duration",
                "BounceRates",
                "ExitRates",
                "PageValues",
                "SpecialDay",
            ],
        )
        fig = px.histogram(raw_df, x=feature, nbins=40, color=target.map({0: "No Purchase", 1: "Purchase"}), title=f"Raw Distribution: {feature}")
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK), legend_title_text="Target")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Raw Data Notes")
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Observed pattern</h4>
                <div class='step-text'>The target class is imbalanced, so recall is more important than accuracy alone.</div>
            </div>
            <br>
            <div class='mini-card'>
                <h4>Why encode month cyclically?</h4>
                <div class='step-text'>Month is seasonal, so December should be close to January, not far away.</div>
            </div>
            <br>
            <div class='mini-card'>
                <h4>Why one-hot categorical fields?</h4>
                <div class='step-text'>Operating system, browser, region, and traffic type are categories, even when stored as numbers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("### Processed Feature Sample")
    st.dataframe(processed_df.head(12), use_container_width=True)

    st.markdown("### Dataset Summary")
    summary = numeric_summary(raw_df)
    summary_fig = px.bar(summary, x="feature", y="mean", title="Average Raw Numeric Feature Values")
    summary_fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
    st.plotly_chart(summary_fig, use_container_width=True)


def render_prediction_studio(raw_df: pd.DataFrame, model: LogisticRegression, preprocessor, metrics: dict) -> None:
    make_title(
        "Prediction Studio",
        "Try a live purchase-intent score using the same feature engineering idea as the project pipeline.",
    )

    defaults = build_input_defaults(raw_df)
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            administrative = st.number_input("Administrative pages", min_value=0, max_value=100, value=int(defaults["Administrative"]), step=1)
            administrative_duration = st.number_input("Administrative duration", min_value=0.0, value=float(defaults["Administrative_Duration"]), step=1.0)
            informational = st.number_input("Informational pages", min_value=0, max_value=50, value=int(defaults["Informational"]), step=1)
            informational_duration = st.number_input("Informational duration", min_value=0.0, value=float(defaults["Informational_Duration"]), step=1.0)
        with col2:
            product_related = st.number_input("Product-related pages", min_value=0, max_value=300, value=int(defaults["ProductRelated"]), step=1)
            product_related_duration = st.number_input("Product-related duration", min_value=0.0, value=float(defaults["ProductRelated_Duration"]), step=1.0)
            bounce_rates = st.slider("Bounce rate", 0.0, 1.0, float(defaults["BounceRates"]), 0.01)
            exit_rates = st.slider("Exit rate", 0.0, 1.0, float(defaults["ExitRates"]), 0.01)
        with col3:
            page_values = st.number_input("Page value", min_value=0.0, value=float(defaults["PageValues"]), step=1.0)
            special_day = st.slider("Special day closeness", 0.0, 1.0, float(defaults["SpecialDay"]), 0.01)
            month = st.selectbox("Month", MONTH_OPTIONS, index=MONTH_OPTIONS.index(defaults["Month"]))
            weekend = st.selectbox("Weekend", [0, 1], index=int(defaults["Weekend"]))

        c1, c2, c3 = st.columns(3)
        with c1:
            operating_system = st.selectbox("Operating system", sorted(raw_df["OperatingSystems"].unique().tolist()), index=0)
            browser = st.selectbox("Browser", sorted(raw_df["Browser"].unique().tolist()), index=0)
        with c2:
            region = st.selectbox("Region", sorted(raw_df["Region"].unique().tolist()), index=0)
            traffic_type = st.selectbox("Traffic type", sorted(raw_df["TrafficType"].unique().tolist()), index=0)
        with c3:
            visitor_type = st.selectbox("Visitor type", sorted(raw_df["VisitorType"].unique().tolist()), index=0)
            threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

        submitted = st.form_submit_button("Predict purchase intent")

    if submitted:
        input_values = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
            "Month": month,
        }
        feature_frame = assemble_prediction_frame(input_values)
        transformed = preprocessor.transform(feature_frame)
        score = float(model.predict_proba(transformed)[0, 1])
        decision = int(score >= threshold)

        res_col1, res_col2 = st.columns([0.9, 1.1])
        with res_col1:
            st.markdown(
                f"""
                <div class='card'>
                    <div class='metric-label'>Purchase probability</div>
                    <div class='metric-value'>{score:.2%}</div>
                    <div class='metric-note'>Threshold: {threshold:.0%} | Decision: {'Purchase' if decision else 'No Purchase'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": TEAL},
                        "steps": [
                            {"range": [0, 40], "color": "#e2e8f0"},
                            {"range": [40, 70], "color": SOFT_AMBER},
                            {"range": [70, 100], "color": SOFT_GREEN},
                        ],
                        "threshold": {"line": {"color": AMBER, "width": 4}, "value": threshold * 100},
                    },
                    title={"text": "Intent Score"},
                )
            )
            gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
            st.plotly_chart(gauge, use_container_width=True)
        with res_col2:
            st.markdown("### Demo Model Quality")
            st.markdown(
                f"""
                <div class='mini-card' style='margin-bottom:0.8rem;'>
                    <h4>Accuracy</h4>
                    <div class='step-text'>{metrics['accuracy']:.3f}</div>
                </div>
                <div class='mini-card' style='margin-bottom:0.8rem;'>
                    <h4>Recall</h4>
                    <div class='step-text'>{metrics['recall']:.3f}</div>
                </div>
                <div class='mini-card' style='margin-bottom:0.8rem;'>
                    <h4>F1 Score</h4>
                    <div class='step-text'>{metrics['f1']:.3f}</div>
                </div>
                <div class='mini-card'>
                    <h4>ROC-AUC</h4>
                    <div class='step-text'>{metrics['roc_auc']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cm = metrics["confusion_matrix"]
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=[SOFT_BLUE, TEAL],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["No Purchase", "Purchase"],
                y=["No Purchase", "Purchase"],
                title="Confusion Matrix",
            )
            cm_fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
            st.plotly_chart(cm_fig, use_container_width=True)

        st.markdown("### Top Coefficients Driving the Demo Model")
        coef_df = metrics["top_coefficients"].reset_index()
        coef_df.columns = ["Feature", "Coefficient"]
        coef_fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="Most Influential Processed Features")
        coef_fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK))
        st.plotly_chart(coef_fig, use_container_width=True)

        st.markdown("### Input Snapshot")
        st.dataframe(feature_frame, use_container_width=True)


def render_serving_hub(raw_df: pd.DataFrame, model: LogisticRegression, preprocessor) -> None:
    make_title(
        "Serving Hub",
        "Production-style serving interface: real-time prediction, decision policy, and batch inference with exportable results.",
    )

    st.markdown("### Serving Modes")
    rt_col, batch_col, policy_col = st.columns(3)
    with rt_col:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Real-time API</h4>
                <div class='step-text'>One visitor session in, one purchase intent score out, with millisecond-level latency.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with batch_col:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Batch Inference</h4>
                <div class='step-text'>Upload multiple sessions, score all, and download a result file for campaign operations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with policy_col:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Decision Logic</h4>
                <div class='step-text'>Apply a probability threshold to trigger actions such as discount, personalization, or no action.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["Real-time Request", "Batch Scoring", "API Contract"])

    with tabs[0]:
        st.markdown("#### Real-time Request Simulator")
        defaults = build_input_defaults(raw_df)
        threshold = st.slider("Serving threshold", 0.05, 0.95, 0.50, 0.01, key="serve_threshold")

        c1, c2, c3 = st.columns(3)
        with c1:
            administrative = st.number_input("Administrative pages", 0, 100, int(defaults["Administrative"]), 1, key="s_admin")
            administrative_duration = st.number_input("Administrative duration", 0.0, value=float(defaults["Administrative_Duration"]), step=1.0, key="s_admin_d")
            informational = st.number_input("Informational pages", 0, 50, int(defaults["Informational"]), 1, key="s_info")
            informational_duration = st.number_input("Informational duration", 0.0, value=float(defaults["Informational_Duration"]), step=1.0, key="s_info_d")
        with c2:
            product_related = st.number_input("Product pages", 0, 300, int(defaults["ProductRelated"]), 1, key="s_prod")
            product_related_duration = st.number_input("Product duration", 0.0, value=float(defaults["ProductRelated_Duration"]), step=1.0, key="s_prod_d")
            bounce_rates = st.slider("Bounce rate", 0.0, 1.0, float(defaults["BounceRates"]), 0.01, key="s_bounce")
            exit_rates = st.slider("Exit rate", 0.0, 1.0, float(defaults["ExitRates"]), 0.01, key="s_exit")
        with c3:
            page_values = st.number_input("Page value", 0.0, value=float(defaults["PageValues"]), step=1.0, key="s_page")
            special_day = st.slider("Special day", 0.0, 1.0, float(defaults["SpecialDay"]), 0.01, key="s_special")
            month = st.selectbox("Month", MONTH_OPTIONS, index=MONTH_OPTIONS.index(defaults["Month"]), key="s_month")
            weekend = st.selectbox("Weekend", [0, 1], index=int(defaults["Weekend"]), key="s_weekend")

        m1, m2, m3 = st.columns(3)
        with m1:
            operating_system = st.selectbox("Operating system", sorted(raw_df["OperatingSystems"].unique().tolist()), key="s_os")
            browser = st.selectbox("Browser", sorted(raw_df["Browser"].unique().tolist()), key="s_browser")
        with m2:
            region = st.selectbox("Region", sorted(raw_df["Region"].unique().tolist()), key="s_region")
            traffic_type = st.selectbox("Traffic type", sorted(raw_df["TrafficType"].unique().tolist()), key="s_traffic")
        with m3:
            visitor_type = st.selectbox("Visitor type", sorted(raw_df["VisitorType"].unique().tolist()), key="s_visitor")
            run_rt = st.button("Run real-time inference", type="primary")

        if run_rt:
            payload = {
                "Administrative": administrative,
                "Administrative_Duration": administrative_duration,
                "Informational": informational,
                "Informational_Duration": informational_duration,
                "ProductRelated": product_related,
                "ProductRelated_Duration": product_related_duration,
                "BounceRates": bounce_rates,
                "ExitRates": exit_rates,
                "PageValues": page_values,
                "SpecialDay": special_day,
                "OperatingSystems": operating_system,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic_type,
                "VisitorType": visitor_type,
                "Weekend": weekend,
                "Month": month,
            }
            feature_frame = assemble_prediction_frame(payload)

            t0 = time.perf_counter()
            transformed = preprocessor.transform(feature_frame)
            score = float(model.predict_proba(transformed)[0, 1])
            decision = int(score >= threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Purchase probability", f"{score:.2%}")
            with r2:
                st.metric("Decision", "Purchase" if decision else "No Purchase")
            with r3:
                st.metric("Latency", f"{latency_ms:.2f} ms")

            response = {
                "purchase_probability": round(score, 6),
                "decision": "purchase" if decision else "no_purchase",
                "threshold": round(float(threshold), 2),
                "latency_ms": round(float(latency_ms), 3),
            }
            st.markdown("##### API-style response")
            st.code(json.dumps(response, indent=2), language="json")

    with tabs[1]:
        st.markdown("#### Batch Scoring")
        st.write("Upload a CSV with raw serving columns and get predictions for all rows.")
        st.caption("Required columns: " + ", ".join(SERVING_RAW_INPUT_COLUMNS))

        uploaded = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_upload")
        threshold_batch = st.slider("Batch decision threshold", 0.05, 0.95, 0.50, 0.01, key="batch_threshold")

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                serving_frame = prepare_serving_frame_from_raw(batch_df)

                t0 = time.perf_counter()
                transformed = preprocessor.transform(serving_frame)
                proba = model.predict_proba(transformed)[:, 1]
                latency_ms = (time.perf_counter() - t0) * 1000.0

                result_df = batch_df.copy()
                result_df["purchase_probability"] = proba
                result_df["decision"] = np.where(proba >= threshold_batch, "purchase", "no_purchase")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Rows scored", f"{len(result_df):,}")
                with c2:
                    st.metric("Avg probability", f"{result_df['purchase_probability'].mean():.2%}")
                with c3:
                    st.metric("Batch latency", f"{latency_ms:.2f} ms")

                st.dataframe(result_df.head(20), use_container_width=True)
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored results",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")

    with tabs[2]:
        st.markdown("#### API Contract (example)")
        request_example = {
            "Administrative": 2,
            "Administrative_Duration": 35.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 28,
            "ProductRelated_Duration": 980.0,
            "BounceRates": 0.01,
            "ExitRates": 0.04,
            "PageValues": 20.0,
            "SpecialDay": 0.0,
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": "Returning_Visitor",
            "Weekend": 0,
            "Month": "Nov",
        }
        response_example = {
            "purchase_probability": 0.74231,
            "decision": "purchase",
            "threshold": 0.5,
            "latency_ms": 7.12,
        }
        st.markdown("**POST /predict**")
        st.code(json.dumps(request_example, indent=2), language="json")
        st.markdown("**Response 200**")
        st.code(json.dumps(response_example, indent=2), language="json")
        st.markdown("**Decision policy**")
        st.code(
            "if purchase_probability >= threshold: action = 'targeted_offer'\nelse: action = 'no_intervention'",
            language="python",
        )



def render_monitoring_center(metadata: dict, drift_report: pd.DataFrame, metrics: dict) -> None:
    make_title(
        "Monitoring Center",
        "Track drift, model quality, artifacts, and the retraining loop in one place.",
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Drift Summary")
        st.dataframe(drift_report, use_container_width=True, height=360)
    with right:
        st.markdown("### What the drift report means")
        st.markdown(
            """
            <div class='mini-card' style='margin-bottom:0.8rem;'>
                <h4>PSI above 0.2</h4>
                <div class='step-text'>Usually considered a meaningful change in distribution.</div>
            </div>
            <div class='mini-card' style='margin-bottom:0.8rem;'>
                <h4>Model monitoring</h4>
                <div class='step-text'>If drift grows or business KPIs fall, the pipeline should retrain from a newer data version.</div>
            </div>
            <div class='mini-card'>
                <h4>Current version</h4>
                <div class='step-text'>The current version in this project is <span class='mono'>v1</span>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Artifact Inventory")
    artifact_cols = st.columns(5)
    artifact_list = [
        ("preprocessor.joblib", "Reusable transformation pipeline"),
        ("X_processed.csv", "Encoded and scaled features"),
        ("y.csv", "Target labels"),
        ("drift_report.csv", "Feature drift monitoring"),
        ("metadata.json", "Version summary and stats"),
    ]
    for col, (fname, desc) in zip(artifact_cols, artifact_list):
        with col:
            st.markdown(
                f"""
                <div class='mini-card'>
                    <h4>{fname}</h4>
                    <div class='step-text'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Monitoring KPI Panel")
    k1, k2, k3, k4 = st.columns(4)
    values = [
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    ]
    for col, (name, value) in zip([k1, k2, k3, k4], values):
        with col:
            st.markdown(
                f"""
                <div class='card'>
                    <div class='metric-label'>{name}</div>
                    <div class='metric-value'>{value:.3f}</div>
                    <div class='metric-note'>Current demo model score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Retraining Loop")
    loop_cols = st.columns(3)
    loop_cards = [
        ("Detect", "Watch PSI and KPI changes."),
        ("Retrain", "Use a newer data version when drift is high."),
        ("Redeploy", "Push the new model and keep monitoring."),
    ]
    for col, (title, desc) in zip(loop_cols, loop_cards):
        with col:
            st.markdown(
                f"""
                <div class='mini-card'>
                    <h4>{title}</h4>
                    <div class='step-text'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_deploy_page(metadata: dict) -> None:
    make_title(
        "Deploy the Website",
        "This project is now a deployable website, not only a localhost app. Use the GitHub repo and Streamlit Cloud or another hosting service.",
    )
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("### Best deployment path")
        st.markdown(
            """
            <div class='mini-card' style='margin-bottom:0.8rem;'>
                <h4>Streamlit Community Cloud</h4>
                <div class='step-text'>Best for this project because it is already a Streamlit app and your repo is on GitHub.</div>
            </div>
            <div class='mini-card' style='margin-bottom:0.8rem;'>
                <h4>Render / Railway</h4>
                <div class='step-text'>Good alternatives if you want a custom domain or a different hosting setup.</div>
            </div>
            <div class='mini-card'>
                <h4>GitHub connection</h4>
                <div class='step-text'>Use the repository <span class='mono'>https://github.com/rahiatkiamona/MLSD-PROJECT</span> as the source.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Deploy steps")
        deploy_steps = [
            "1. Push the latest code to GitHub.",
            "2. Open Streamlit Community Cloud.",
            "3. Choose the repo and branch.",
            "4. Set the main file as app.py.",
            "5. Click Deploy and wait for the URL.",
        ]
        for step in deploy_steps:
            st.markdown(f"- {step}")
        st.info(
            "If you want a more polished public deployment, I can also create a Docker setup or a Render-ready configuration next."
        )

    st.markdown("### What users will see online")
    st.write(
        "A live website with a sidebar navigation, data explorer, prediction studio, monitoring center, and deployment guide."
    )
    st.write(
        f"Project version: {metadata['data_version']} | Rows: {metadata['total_rows']:,} | Features after encoding: {metadata['total_features_after_encoding']}"
    )


def main() -> None:
    inject_styles()
    metadata = load_metadata()
    drift_report = load_drift_report()
    raw_df = load_raw_data()
    processed_df = load_processed_data()
    target = load_target()

    clean_frame = clean_training_frame(raw_df)
    preprocessor = build_preprocessor(clean_frame)
    processed_matrix = preprocessor.transform(clean_frame)
    processed_frame = pd.DataFrame(processed_matrix, columns=load_processed_data().columns, index=clean_frame.index)

    model, metrics, _ = train_demo_model(processed_frame, target)
    page = sidebar_nav(metadata, metrics)

    if page == "Home":
        render_home(metadata, drift_report, raw_df, processed_frame, target, metrics)
    elif page == "Data Lab":
        render_data_lab(raw_df, processed_frame, target)
    elif page == "Prediction Studio":
        render_prediction_studio(raw_df, model, preprocessor, metrics)
    elif page == "Serving Hub":
        render_serving_hub(raw_df, model, preprocessor)
    elif page == "Monitoring Center":
        render_monitoring_center(metadata, drift_report, metrics)
    elif page == "Deploy":
        render_deploy_page(metadata)


if __name__ == "__main__":
    main()
