"""
Flight Price Predictor — Streamlit UI

Runs the prediction pipeline directly (no separate API server needed),
so it deploys cleanly to Streamlit Cloud.
"""

import datetime
import os

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* overall background */
        .stApp { background-color: #f7f9fc; }

        /* card container */
        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 28px 32px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            margin-bottom: 24px;
        }

        /* price result */
        .price-box {
            background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
            border-radius: 16px;
            padding: 36px 24px;
            text-align: center;
            color: white;
        }
        .price-label  { font-size: 1rem;  opacity: 0.85; letter-spacing: 0.05em; }
        .price-amount { font-size: 3rem;  font-weight: 700; margin: 8px 0; }
        .price-sub    { font-size: 0.85rem; opacity: 0.75; }

        /* section headers */
        .section-title {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        /* mute the default st.form border */
        .stForm { border: none !important; }

        /* hide streamlit footer */
        footer { visibility: hidden; }

        /* metric cards */
        div[data-testid="metric-container"] {
            background: #f0f4ff;
            border-radius: 10px;
            padding: 12px 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model loader (cached so it only loads once) ───────────────────────────────
DAGSHUB_REPO = "ahmedsh711/mlops-flight-predictor"
DAGSHUB_MODEL_URL = (
    f"https://dagshub.com/{DAGSHUB_REPO}/raw/main/models/pipeline.joblib"
)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib

    from flight_predictor.config import MODELS_DIR, PIPELINE_PATH

    # If artifact already exists locally (local dev or if committed), use it
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)

    # On Streamlit Cloud the model isn't in git (DVC tracks it), so we
    # try to pull it from DagsHub. Requires DAGSHUB_TOKEN in st.secrets.
    token = st.secrets.get("DAGSHUB_TOKEN", os.getenv("DAGSHUB_TOKEN", ""))
    if not token:
        return None

    try:
        import io

        import requests

        headers = {"Authorization": f"token {token}"}
        resp = requests.get(DAGSHUB_MODEL_URL, headers=headers, timeout=30)
        resp.raise_for_status()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_bytes = io.BytesIO(resp.content)
        pipeline = joblib.load(model_bytes)

        # Cache to disk so subsequent calls skip the download
        joblib.dump(pipeline, PIPELINE_PATH)
        return pipeline
    except Exception as e:
        st.warning(f"Could not fetch model from DagsHub: {e}", icon="⚠️")
        return None


@st.cache_data(show_spinner=False)
def get_model_info():
    import json

    from flight_predictor.config import MODEL_INFO_PATH

    if not MODEL_INFO_PATH.exists():
        return None
    with open(MODEL_INFO_PATH) as f:
        return json.load(f)


def predict_price(pipeline, form_data: dict) -> float:
    import pandas as pd

    from flight_predictor.features import engineer_features, get_feature_columns

    df = pd.DataFrame([form_data])
    df_eng = engineer_features(df)
    X = df_eng[get_feature_columns()]
    return float(pipeline.predict(X)[0])


# ── City → airport code mapping ───────────────────────────────────────────────
CITY_TO_CODE = {
    "Bangalore": "BLR",
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Chennai": "MAA",
    "Kolkata": "CCU",
    "Cochin": "COK",
    "Hyderabad": "HYD",
}

AIRLINES = [
    "IndiGo",
    "Air India",
    "Jet Airways",
    "SpiceJet",
    "GoAir",
    "Vistara",
    "Air Asia",
    "Other",
]

CITIES = list(CITY_TO_CODE.keys())

STOPS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]

STOPS_EMOJI = {
    "non-stop": "🟢",
    "1 stop": "🟡",
    "2 stops": "🟠",
    "3 stops": "🔴",
    "4 stops": "🔴",
}

AIRLINE_LOGO = {
    "IndiGo": "💙",
    "Air India": "🧡",
    "Jet Airways": "🟡",
    "SpiceJet": "🌶️",
    "GoAir": "🟢",
    "Vistara": "🔵",
    "Air Asia": "❤️",
    "Other": "✈️",
}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ✈️ Flight Price Predictor")
st.markdown(
    "Get an instant price estimate for Indian domestic flights powered by an XGBoost model "
    "trained on real booking data."
)
st.divider()


# ── Load model & show status ──────────────────────────────────────────────────
pipeline = load_model()
model_info = get_model_info()

if pipeline is None:
    st.error(
        "**Model not loaded.** The `pipeline.joblib` artifact was not found.",
        icon="🚨",
    )
    st.info(
        "**Running on Streamlit Cloud?** Add your DagsHub token as a secret:\n\n"
        "1. Go to **App settings → Secrets** in Streamlit Cloud\n"
        "2. Add `DAGSHUB_TOKEN = \"your_token_here\"`\n"
        "3. Restart the app — the model will be fetched automatically.",
        icon="🔑",
    )
    st.stop()


# ── Main form ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    # Flight details
    st.markdown('<p class="section-title">Flight details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox(
            "Airline",
            AIRLINES,
            format_func=lambda x: f"{AIRLINE_LOGO.get(x, '✈️')}  {x}",
        )
        source = st.selectbox("From", CITIES, index=1)  # Delhi default
    with col2:
        total_stops = st.selectbox(
            "Stops",
            STOPS,
            format_func=lambda x: f"{STOPS_EMOJI[x]}  {x}",
        )
        destination = st.selectbox("To", CITIES, index=0)  # Bangalore default

    st.markdown('<p class="section-title" style="margin-top:16px">Date & time</p>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns([2, 1, 1])
    with col3:
        journey_date = st.date_input(
            "Date of journey",
            value=datetime.date.today() + datetime.timedelta(days=7),
            min_value=datetime.date.today(),
        )
    with col4:
        dep_time = st.time_input("Departure", value=datetime.time(6, 0), step=300)
    with col5:
        arr_time = st.time_input("Arrival", value=datetime.time(8, 30), step=300)

    st.markdown('<p class="section-title" style="margin-top:16px">Duration</p>', unsafe_allow_html=True)

    dur_col1, dur_col2, _ = st.columns([1, 1, 2])
    with dur_col1:
        dur_hours = st.number_input("Hours", min_value=0, max_value=24, value=2, step=1)
    with dur_col2:
        dur_mins = st.number_input("Minutes", min_value=0, max_value=59, value=30, step=5)

    additional_info = st.selectbox(
        "Additional info",
        ["No info", "In-flight meal included", "No check-in baggage included", "Red-eye flight"],
        index=0,
    )

    submitted = st.form_submit_button(
        "🔍  Predict price",
        use_container_width=True,
        type="primary",
    )


# ── Run prediction ────────────────────────────────────────────────────────────
if submitted:
    # Validate source ≠ destination
    if source == destination:
        st.warning("Source and destination can't be the same city.", icon="⚠️")
        st.stop()

    # Build duration string
    if dur_hours > 0 and dur_mins > 0:
        duration_str = f"{dur_hours}h {dur_mins}m"
    elif dur_hours > 0:
        duration_str = f"{dur_hours}h"
    else:
        duration_str = f"{dur_mins}m"

    form_data = {
        "Airline": airline,
        "Date_of_Journey": journey_date.strftime("%d/%m/%Y"),
        "Source": source,
        "Destination": destination,
        "Route": f"{CITY_TO_CODE[source]} → {CITY_TO_CODE[destination]}",
        "Dep_Time": dep_time.strftime("%H:%M"),
        "Arrival_Time": arr_time.strftime("%H:%M"),
        "Duration": duration_str,
        "Total_Stops": total_stops,
        "Additional_Info": additional_info,
    }

    with st.spinner("Calculating…"):
        try:
            price = predict_price(pipeline, form_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}", icon="🚨")
            st.stop()

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown("---")

    st.markdown(
        f"""
        <div class="price-box">
            <p class="price-label">ESTIMATED PRICE</p>
            <p class="price-amount">₹{price:,.0f}</p>
            <p class="price-sub">{airline} &nbsp;·&nbsp; {source} → {destination} &nbsp;·&nbsp; {total_stops}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Flight summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Departure", dep_time.strftime("%H:%M"))
    c2.metric("Arrival", arr_time.strftime("%H:%M"))
    c3.metric("Duration", duration_str)

    st.markdown("")

    # Per-class rough estimate note
    if price < 4000:
        tier = "Economy / budget"
        note = "Looks like a competitive price — worth booking early."
    elif price < 10000:
        tier = "Mid-range"
        note = "Fairly average for this route. Prices may vary with demand."
    else:
        tier = "Premium / peak-season"
        note = "High estimate — consider flexible dates or a different airline."

    st.info(f"**{tier}** — {note}", icon="💡")

    # Model confidence footnote
    if model_info:
        r2 = model_info.get("metrics", {}).get("r2_inr_space", 0)
        rmse = model_info.get("metrics", {}).get("rmse_inr", 0)
        st.caption(
            f"Model R² = {r2:.3f} · RMSE ≈ ₹{rmse:,.0f} on held-out test data "
            f"(XGBoost + TransformedTargetRegressor, log₁p target)"
        )


# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About the model")
    st.markdown(
        "This app uses an **XGBoost regressor** wrapped in a "
        "`TransformedTargetRegressor` (log₁p / expm1). "
        "Trained on the Kaggle Indian domestic flights dataset."
    )

    if model_info:
        st.divider()
        st.markdown("**Model metrics (test set)**")
        metrics = model_info.get("metrics", {})
        st.metric("R²", f"{metrics.get('r2_inr_space', 0):.4f}")
        st.metric("RMSE", f"₹{metrics.get('rmse_inr', 0):,.0f}")
        st.metric("MAE", f"₹{metrics.get('mae_inr', 0):,.0f}")

        st.divider()
        st.markdown("**Training info**")
        st.caption(f"Features: {model_info.get('feature_count', '—')}")
        st.caption(f"Train samples: {model_info.get('train_size', '—'):,}")
        st.caption(f"CV folds: 5")
        cv_mean = model_info.get("cv_mean_r2", 0)
        cv_std = model_info.get("cv_std_r2", 0)
        st.caption(f"CV R² = {cv_mean:.4f} ± {cv_std:.4f}")

    st.divider()
    st.markdown(
        "[GitHub](https://github.com/ahmedsh711/mlops-flight-predictor) · "
        "Built with Streamlit"
    )
