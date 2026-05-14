from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Online Shoppers | Purchase Intent Dashboard",
    page_icon="shopping-cart",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "v1"
RAW_DATA_PATH = ROOT / "online_shoppers_intention.csv"


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


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load(ARTIFACT_DIR / "preprocessor.joblib")


metadata = load_metadata()
drift_report = load_drift_report()
raw_df = load_raw_data()
processed_df = load_processed_data()
target = load_target()
_ = load_preprocessor()

positive_count = int(target.sum())
negative_count = int(len(target) - positive_count)
positive_rate = float(metadata["positive_rate"])
high_drift_count = int(metadata["features_with_high_drift"])

calm_bg = "#f7f5ef"
ink = "#0f172a"
teal = "#0f766e"
muted_teal = "#5fa8a8"
amber = "#c28b2c"
soft_blue = "#dbeafe"
soft_green = "#d1fae5"
soft_amber = "#fef3c7"
soft_slate = "#e2e8f0"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {calm_bg} 0%, #ffffff 35%, #f8fafc 100%);
        color: {ink};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}
    h1, h2, h3, h4, h5 {{
        color: {ink};
        letter-spacing: -0.02em;
    }}
    .hero {{
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 28px;
        background: radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 35%),
                    linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,255,255,0.75));
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
        font-size: 1.05rem;
        line-height: 1.65;
    }}
    .pill {{
        display: inline-block;
        padding: 0.42rem 0.85rem;
        border-radius: 999px;
        background: rgba(15,118,110,0.10);
        color: {teal};
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
    }}
    .card {{
        border-radius: 22px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(148,163,184,0.18);
        padding: 1.1rem 1.1rem 1rem 1.1rem;
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
        color: {ink};
        line-height: 1.1;
    }}
    .metric-note {{
        color: #475569;
        margin-top: 0.35rem;
        font-size: 0.94rem;
    }}
    .step-grid {{
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 0.8rem;
    }}
    .step {{
        border-radius: 18px;
        padding: 0.9rem;
        background: white;
        border: 1px solid rgba(148,163,184,0.18);
        min-height: 128px;
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
        color: {teal};
    }}
    .step-title {{
        font-weight: 800;
        margin-bottom: 0.35rem;
        color: {ink};
    }}
    .step-text {{
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.45;
    }}
    .section-title {{
        margin-top: 1rem;
        margin-bottom: 0.6rem;
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
    .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.9rem;
        background: #f8fafc;
        padding: 0.15rem 0.35rem;
        border-radius: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="pill">Versioned ML System</div>
        <div class="pill">Purchase Intent Prediction</div>
        <div class="pill">Recall + F1 Focus</div>
        <h1>Online Shoppers Purchase Intent Dashboard</h1>
        <p class="subtle">
            A calm, complete visual of your project: raw data ingestion, cleaning, feature engineering,
            encoding, drift monitoring, ML training, serving, and retraining. This is the project story
            your team can understand at a glance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Rows</div>
            <div class="metric-value">{metadata['total_rows']:,}</div>
            <div class="metric-note">Sessions in the versioned dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{metadata['total_features_after_encoding']}</div>
            <div class="metric-note">Model-ready columns after encoding</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Positive Rate</div>
            <div class="metric-value">{positive_rate:.2f}%</div>
            <div class="metric-note">Users who purchased in the dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">High Drift Features</div>
            <div class="metric-value">{high_drift_count}</div>
            <div class="metric-note">Features flagged by PSI drift check</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Project Meaning")
st.write(
    "This project predicts whether an online visitor will purchase, so business teams can trigger targeted actions, "
    "personalized recommendations, and better marketing spend. The main business value is higher revenue, better ROI, "
    "and fewer wasted impressions."
)

left, right = st.columns([1.2, 1])
with left:
    st.markdown("### End-to-End Flow")
    steps = [
        ("1", "Raw Data", "CSV / DB source from the Kaggle dataset."),
        ("2", "Versioned Ingestion", "Save each data revision as v1, v2, ..."),
        ("3", "Data Pipeline", "Validate, clean, engineer, encode, and monitor drift."),
        ("4", "ML Pipeline", "Split data, train models, tune, and evaluate recall/F1."),
        ("5", "Serving Pipeline", "Run real-time or batch inference with a threshold."),
        ("6", "Monitoring", "Track business KPIs, latency, drift, and retraining triggers."),
        ("7", "Artifacts", "Store processed features, metadata, and preprocessor objects."),
    ]
    row1 = st.columns(4)
    row2 = st.columns(3)
    for col, step in zip(row1, steps[:4]):
        num, title, text = step
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="min-height: 145px;">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    for col, step in zip(row2, steps[4:]):
        num, title, text = step
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="min-height: 145px;">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    st.markdown("### Why These Metrics")
    metric_cards = """
    <div class='mini-card' style='margin-bottom:0.9rem;'>
        <h4>Recall</h4>
        <div class='step-text'>Catching potential buyers is more important than missing them. A false positive is usually cheaper than a missed purchase.</div>
    </div>
    <div class='mini-card' style='margin-bottom:0.9rem;'>
        <h4>F1 Score</h4>
        <div class='step-text'>Balances precision and recall so the model does not over-predict buyers or under-predict them.</div>
    </div>
    <div class='mini-card'>
        <h4>Latency</h4>
        <div class='step-text'>For live targeting, prediction must be fast so the visitor can still be influenced while browsing.</div>
    </div>
    """
    st.markdown(metric_cards, unsafe_allow_html=True)

st.markdown("### Data Story")
st.write(
    "The raw dataset contains user session behavior, page visits, timing signals, and browsing context. "
    "The pipeline cleans the data, builds richer behavioral features, converts categorical information into a usable model format, "
    "and saves versioned outputs so the same transformation can be reused later."
)

c5, c6 = st.columns(2)
with c5:
    target_fig = go.Figure(
        data=[
            go.Pie(
                labels=["No Purchase", "Purchase"],
                values=[negative_count, positive_count],
                hole=0.58,
                marker=dict(colors=["#cbd5e1", teal]),
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
        font=dict(color=ink),
    )
    st.plotly_chart(target_fig, use_container_width=True)

with c6:
    drift_plot = drift_report.sort_values("psi", ascending=True).tail(10)
    drift_fig = px.bar(
        drift_plot,
        x="psi",
        y="feature",
        orientation="h",
        color="drift_flag",
        color_discrete_map={True: "#c28b2c", False: teal},
        title="Top Drift Features (PSI)",
    )
    drift_fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink),
        xaxis_title="PSI",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(drift_fig, use_container_width=True)

st.markdown("### Feature Engineering")
feature_cols = st.columns(5)
feature_cards = [
    ("TotalSessionDuration", "Sum of all page durations."),
    ("ProductInfoRatio", "Product focus vs informational browsing."),
    ("EngagementScore", "Value relative to time spent."),
    ("Month_sin", "Seasonality encoded as a cycle."),
    ("Month_cos", "Seasonality encoded as a cycle."),
]
for col, (name, desc) in zip(feature_cols, feature_cards):
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

st.markdown("### Pipeline Decisions")
pipe_left, pipe_mid, pipe_right = st.columns(3)
with pipe_left:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Numerical Pipeline</h4>
            <div class='step-text'>Median imputation then standardization keeps numeric features stable for ML algorithms.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with pipe_mid:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Categorical Pipeline</h4>
            <div class='step-text'>One-hot encoding converts browser, region, OS, traffic type, and visitor type into machine-friendly inputs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with pipe_right:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Versioning</h4>
            <div class='step-text'>The project saves each processing version so training and serving can always use the same data logic.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Saved Artifacts")
artifact_names = [
    ("preprocessor.joblib", "Reusable transformation pipeline"),
    ("X_processed.csv", "Encoded and scaled features"),
    ("y.csv", "Target labels"),
    ("drift_report.csv", "Feature drift monitoring"),
    ("metadata.json", "Version summary and stats"),
]
cols = st.columns(5)
for col, (fname, desc) in zip(cols, artifact_names):
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

st.markdown("### Business Actions")
a1, a2, a3 = st.columns(3)
with a1:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Targeted Offers</h4>
            <div class='step-text'>Send discounts or nudges to high-probability users before they leave.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with a2:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Personalization</h4>
            <div class='step-text'>Adapt product recommendations and landing page content to visitor behavior.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with a3:
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Monitoring Loop</h4>
            <div class='step-text'>If drift or KPIs degrade, retrain using a newer data version.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Project summary for teammates", expanded=False):
    st.write(
        "This dashboard shows the full system architecture: raw data enters a versioned pipeline, "
        "the data is cleaned and engineered, encoded features are saved, a model is trained and evaluated, "
        "and the service is monitored so drift or business KPI drops can trigger retraining."
    )
    st.write(
        f"Current artifact version: {metadata['data_version']} | Total rows: {metadata['total_rows']:,} | "
        f"Output features: {metadata['total_features_after_encoding']} | High-drift features: {high_drift_count}"
    )
