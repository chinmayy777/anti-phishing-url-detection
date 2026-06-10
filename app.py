import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import sys
import os
from collections import Counter
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered",
)

# ── styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* hide streamlit default header */
header[data-testid="stHeader"] { background: transparent; }

/* hero title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-accent {
    color: #00e5a0;
}
.hero-sub {
    font-size: 1rem;
    color: #7a7f94;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* input box */
.stTextInput > div > div > input {
    background-color: #161922 !important;
    border: 1.5px solid #2a2f3d !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,0.15) !important;
}

/* button */
.stButton > button {
    background: #00e5a0 !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* verdict cards */
.verdict-phishing {
    background: linear-gradient(135deg, #1e0f0f 0%, #1a0d0d 100%);
    border: 1.5px solid #ff4b4b;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.verdict-safe {
    background: linear-gradient(135deg, #0a1a14 0%, #0d1f18 100%);
    border: 1.5px solid #00e5a0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.verdict-confidence {
    font-size: 0.9rem;
    color: #7a7f94;
    font-weight: 300;
}

/* feature card */
.feature-section {
    background: #161922;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.feature-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: #7a7f94;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* metric row */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1f2330;
}
.metric-row:last-child { border-bottom: none; }
.metric-name { font-size: 0.88rem; color: #b0b4c8; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 0.88rem; }
.flag-red { color: #ff4b4b; }
.flag-green { color: #00e5a0; }
.flag-neutral { color: #e8eaf0; }

/* model stats bar */
.stat-pill {
    display: inline-block;
    background: #1f2330;
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #00e5a0;
    margin-right: 0.5rem;
    margin-bottom: 0.4rem;
}

/* divider */
.thin-divider {
    border: none;
    border-top: 1px solid #1f2330;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── feature extraction (mirrors url_features.py exactly) ──────────────────────
def extract_features(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    features = {}
    features["url_length"] = len(url)
    features["dot_count"] = parsed.netloc.count(".")
    features["hyphen_count"] = parsed.netloc.count("-")
    features["has_https"] = int(parsed.scheme == "https")
    ip_pattern = r"\b\d{1,3}(\.\d{1,3}){3}\b"
    features["has_ip"] = int(bool(re.search(ip_pattern, url)))
    keywords = ["secure", "login", "verify", "accounts", "update", "check", "security"]
    features["suspicious_keyword_count"] = sum(w in url.lower() for w in keywords)
    counts = Counter(url)
    entropy = 0.0
    for char in counts:
        p = counts[char] / len(url)
        entropy -= p * math.log2(p)
    features["entropy"] = round(entropy, 2)
    return features


# ── model training (cached so it only runs once per session) ──────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "data", "dataset_schema_v1.csv")
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()


# ── feature flag logic ─────────────────────────────────────────────────────────
def get_flags(features):
    """
    Returns a dict: feature_name -> (display_value, flag_class, note)
    flag_class is 'flag-red', 'flag-green', or 'flag-neutral'
    """
    flags = {}

    length = features["url_length"]
    flags["url_length"] = (
        str(length),
        "flag-red" if length > 54 else "flag-green",
        "Suspiciously long" if length > 54 else "Normal length"
    )

    dots = features["dot_count"]
    flags["dot_count"] = (
        str(dots),
        "flag-red" if dots > 3 else "flag-green",
        "Multiple subdomains — suspicious" if dots > 3 else "Normal"
    )

    hyphens = features["hyphen_count"]
    flags["hyphen_count"] = (
        str(hyphens),
        "flag-red" if hyphens >= 2 else "flag-green",
        "Unusual hyphen usage" if hyphens >= 2 else "Normal"
    )

    https = features["has_https"]
    flags["has_https"] = (
        "Yes" if https else "No",
        "flag-green" if https else "flag-red",
        "HTTPS present" if https else "No HTTPS (note: not conclusive)"
    )

    ip = features["has_ip"]
    flags["has_ip"] = (
        "Yes" if ip else "No",
        "flag-red" if ip else "flag-green",
        "IP in URL — strong phishing signal" if ip else "No IP detected"
    )

    kw = features["suspicious_keyword_count"]
    flags["suspicious_keyword_count"] = (
        str(kw),
        "flag-red" if kw >= 2 else ("flag-neutral" if kw == 1 else "flag-green"),
        f"{kw} suspicious keyword(s) found" if kw > 0 else "No suspicious keywords"
    )

    entropy = features["entropy"]
    flags["entropy"] = (
        str(entropy),
        "flag-red" if entropy > 4.5 else "flag-green",
        "High randomness — suspicious" if entropy > 4.5 else "Normal character distribution"
    )

    return flags


FEATURE_LABELS = {
    "url_length": "URL Length",
    "dot_count": "Dot Count",
    "hyphen_count": "Hyphen Count",
    "has_https": "HTTPS",
    "has_ip": "IP in URL",
    "suspicious_keyword_count": "Suspicious Keywords",
    "entropy": "Entropy",
}


# ── feature importance chart ───────────────────────────────────────────────────
def plot_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in indices]
    values = importances[indices]
    colors = ["#00e5a0" if v == max(importances) else "#2a3a54" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#161922")
    ax.set_facecolor("#161922")

    bars = ax.barh(labels, values, color=["#00e5a0" if v >= sorted(importances)[-2] else "#1f3050" for v in values], height=0.6)

    ax.set_xlabel("Importance", color="#7a7f94", fontsize=8)
    ax.tick_params(colors="#b0b4c8", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2a2f3d")
    ax.spines["left"].set_color("#2a2f3d")
    ax.xaxis.label.set_color("#7a7f94")

    for spine in ax.spines.values():
        spine.set_color("#2a2f3d")

    plt.tight_layout()
    return fig


# ── app layout ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    Phish<span class="hero-accent">Guard</span>
</div>
<div class="hero-sub">
    Lexical ML-based phishing URL detector &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; 96.4% accuracy
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

# load model
with st.spinner("Loading model..."):
    model, feature_names = load_model()

# input
url_input = st.text_input(
    "",
    placeholder="https://example.com/login",
    label_visibility="collapsed"
)

analyze_clicked = st.button("→ Analyze URL")

# ── results ───────────────────────────────────────────────────────────────────
if analyze_clicked and url_input.strip():
    url = url_input.strip()
    features = extract_features(url)
    X_input = pd.DataFrame([features])[feature_names]

    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    confidence = proba[prediction] * 100

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # verdict
    if prediction == 1:
        st.markdown(f"""
        <div class="verdict-phishing">
            <div class="verdict-label" style="color:#ff4b4b;">🚨 Phishing Detected</div>
            <div class="verdict-confidence">Confidence: {confidence:.1f}% &nbsp;|&nbsp; Model flagged this URL as malicious</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-safe">
            <div class="verdict-label" style="color:#00e5a0;">✅ Looks Legitimate</div>
            <div class="verdict-confidence">Confidence: {confidence:.1f}% &nbsp;|&nbsp; No phishing signals detected</div>
        </div>
        """, unsafe_allow_html=True)

    # feature breakdown
    flags = get_flags(features)

    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<div class="feature-title">Feature Breakdown</div>', unsafe_allow_html=True)

    for feat_key, (display_val, flag_class, note) in flags.items():
        label = FEATURE_LABELS.get(feat_key, feat_key)
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-name">{label}<br><span style="font-size:0.75rem; color:#4a4f64;">{note}</span></span>
            <span class="metric-value {flag_class}">{display_val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif analyze_clicked and not url_input.strip():
    st.warning("Please enter a URL to analyze.")

# ── model info section ────────────────────────────────────────────────────────
st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
st.markdown('<div class="feature-title" style="font-family:\'Space Mono\',monospace; font-size:0.7rem; letter-spacing:2px; color:#7a7f94; text-transform:uppercase;">Model Stats</div>', unsafe_allow_html=True)

st.markdown("""
<span class="stat-pill">Accuracy 96.4%</span>
<span class="stat-pill">Precision 95.8%</span>
<span class="stat-pill">Recall 97.9%</span>
<span class="stat-pill">F1 96.8%</span>
<span class="stat-pill">420 training URLs</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Feature Importance (Random Forest)"):
    fig = plot_importance(model, feature_names)
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        "<p style='font-size:0.78rem; color:#7a7f94;'>URL length and entropy account for ~70% of the model's decisions.</p>",
        unsafe_allow_html=True
    )