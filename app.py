import streamlit as st
import joblib
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Product Backorder Risk Prediction",
    page_icon="📦",
    layout="wide"
)

# --------------------------------------------------
# 2. Project heading
# --------------------------------------------------

st.title("📦 Product Backorder Risk Prediction")

st.markdown(
    """
    **Machine Learning-Based Inventory Risk Assessment**

    Estimate the likelihood of a product going on backorder
    using inventory levels, supplier lead time, sales forecasts,
    and product risk indicators.
    """
)

st.divider()

# --------------------------------------------------
# 3. Load the saved model and preprocessing details
# --------------------------------------------------

APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    model_path = APP_DIR / "backorder_random_forest.pkl"
    zip_path = APP_DIR / "backorder_random_forest.zip"

    if not model_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("backorder_random_forest.pkl", APP_DIR)

    model = joblib.load(model_path)

    info = joblib.load(
        APP_DIR / "backorder_preprocessing.pkl"
    )

    return model, info


try:
    model, info = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

st.info(
    "Enter product and inventory details below to assess "
    "backorder risk. Model predictions are estimates, "
    "not guarantees."
)

# --------------------------------------------------
# 4. Input form
# --------------------------------------------------

with st.form("backorder_form"):

    st.subheader("Inventory and demand")

    col1, col2, col3 = st.columns(3)

    with col1:
        national_inv = st.number_input(
            "National inventory",
            value=100.0
        )

        lead_time_is_missing = st.checkbox(
            "Lead time is unavailable"
        )

        lead_time = st.number_input(
            "Lead time (days)",
            min_value=0.0,
            value=float(info["train_medians"]["lead_time"]),
            disabled=lead_time_is_missing
        )

        in_transit_qty = st.number_input(
            "In-transit quantity",
            value=20.0
        )

        min_bank = st.number_input(
            "Minimum bank",
            value=0.0
        )

    with col2:
        forecast_3_month = st.number_input(
            "3-month forecast",
            min_value=0.0,
            value=50.0
        )

        forecast_6_month = st.number_input(
            "6-month forecast",
            min_value=0.0,
            value=100.0
        )

        forecast_9_month = st.number_input(
            "9-month forecast",
            min_value=0.0,
            value=150.0
        )

    with col3:
        sales_1_month = st.number_input(
            "1-month sales",
            min_value=0.0,
            value=10.0
        )

        sales_3_month = st.number_input(
            "3-month sales",
            min_value=0.0,
            value=30.0
        )

        sales_6_month = st.number_input(
            "6-month sales",
            min_value=0.0,
            value=60.0
        )

        sales_9_month = st.number_input(
            "9-month sales",
            min_value=0.0,
            value=90.0
        )

    st.subheader("Product and supplier details")

    col4, col5, col6 = st.columns(3)

    with col4:
        pieces_past_due = st.number_input(
            "Pieces past due",
            min_value=0.0,
            value=0.0
        )

        perf_6_month_avg = st.number_input(
            "6-month performance average (-99 = missing)",
            min_value=-99.0,
            max_value=1.0,
            value=0.90,
            step=0.01
        )

        perf_12_month_avg = st.number_input(
            "12-month performance average (-99 = missing)",
            min_value=-99.0,
            max_value=1.0,
            value=0.90,
            step=0.01
        )

    with col5:
        potential_issue = st.selectbox(
            "Potential issue",
            [0, 1]
        )

        deck_risk = st.selectbox(
            "Deck risk",
            [0, 1]
        )

        oe_constraint = st.selectbox(
            "OE constraint",
            [0, 1]
        )

    with col6:
        ppap_risk = st.selectbox(
            "PPAP risk",
            [0, 1]
        )

        stop_auto_buy = st.selectbox(
            "Stop auto buy",
            [0, 1]
        )

        rev_stop = st.selectbox(
            "Revision stop",
            [0, 1]
        )

    submitted = st.form_submit_button(
        "Predict Backorder Risk",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Feature engineering and prediction
# --------------------------------------------------

if submitted:

    # Record missing indicators before imputation
    lead_time_missing = int(lead_time_is_missing)

    perf_6_missing = int(perf_6_month_avg == -99)
    perf_12_missing = int(perf_12_month_avg == -99)

    if lead_time_is_missing:
        lead_time = np.nan

    if perf_6_missing:
        perf_6_month_avg = np.nan

    if perf_12_missing:
        perf_12_month_avg = np.nan

    # Inventory features
    inventory_demand_gap = (
        national_inv - forecast_3_month
    )

    total_available_supply = (
        national_inv + in_transit_qty
    )

    # Supply-to-demand ratio
    supply_demand_ratio = (
        total_available_supply / forecast_3_month
        if forecast_3_month > 0
        else np.nan
    )

    # Sales trends
    sales_trend_1m_vs_3m = (
        (sales_1_month * 3) / sales_3_month
        if sales_3_month > 0
        else np.nan
    )

    sales_trend_3m_vs_6m = (
        (sales_3_month * 2) / sales_6_month
        if sales_6_month > 0
        else np.nan
    )

    # Inventory-to-forecast ratio
    inventory_forecast_ratio = (
        national_inv / forecast_3_month
        if forecast_3_month > 0
        else np.nan
    )

    # Build the feature row
    row = {
        "national_inv": national_inv,
        "lead_time": lead_time,
        "in_transit_qty": in_transit_qty,
        "forecast_3_month": forecast_3_month,
        "forecast_6_month": forecast_6_month,
        "forecast_9_month": forecast_9_month,
        "sales_1_month": sales_1_month,
        "sales_3_month": sales_3_month,
        "sales_6_month": sales_6_month,
        "sales_9_month": sales_9_month,
        "min_bank": min_bank,
        "potential_issue": potential_issue,
        "pieces_past_due": pieces_past_due,
        "perf_6_month_avg": perf_6_month_avg,
        "perf_12_month_avg": perf_12_month_avg,
        "deck_risk": deck_risk,
        "oe_constraint": oe_constraint,
        "ppap_risk": ppap_risk,
        "stop_auto_buy": stop_auto_buy,
        "rev_stop": rev_stop,
        "perf_6_missing": perf_6_missing,
        "perf_12_missing": perf_12_missing,
        "lead_time_missing": lead_time_missing,
        "inventory_demand_gap": inventory_demand_gap,
        "total_available_supply": total_available_supply,
        "supply_demand_ratio": supply_demand_ratio,
        "sales_trend_1m_vs_3m": sales_trend_1m_vs_3m,
        "sales_trend_3m_vs_6m": sales_trend_3m_vs_6m,
        "inventory_forecast_ratio": inventory_forecast_ratio,
        "forecast_3m_zero": int(forecast_3_month == 0),
        "sales_3m_zero": int(sales_3_month == 0),
        "sales_6m_zero": int(sales_6_month == 0),
    }

    # Match training feature names and order
    X_new = pd.DataFrame([row]).reindex(
        columns=info["feature_names"]
    )

    # Apply the saved training medians
    for col in info["impute_cols"]:
        X_new[col] = X_new[col].fillna(
            info["train_medians"][col]
        )

    # Check for remaining missing values
    if X_new.isna().any().any():
        st.error(
            "Missing values remain after preprocessing. "
            "Please check the inputs and preprocessing file."
        )
        st.stop()

    # Predict probability and classification
    probability = model.predict_proba(X_new)[0, 1]

    threshold = info["final_threshold"]
    prediction = int(probability >= threshold)

    # --------------------------------------------------
    # 6. Display prediction results
    # --------------------------------------------------

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        label="Estimated Backorder Probability",
        value=f"{probability:.1%}"
    )

    result_col2.metric(
        label="Decision Threshold",
        value=f"{threshold:.0%}"
    )

    if prediction == 1:
        st.error(
            "⚠️ Higher backorder risk: the model predicts "
            "a backorder."
        )
    else:
        st.success(
            "✅ Lower backorder risk: the model does not "
            "predict a backorder."
        )

    st.caption(
        "The probability is a model estimate, not a guarantee "
        "of the actual product outcome."
    )


it is backorder_random_forest.zip
