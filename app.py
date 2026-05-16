import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pediatric TB Surveillance System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("Pediatric Tuberculosis Surveillance and Early Warning System")

st.markdown("""
### Prototype System

This prototype system was developed to support pediatric tuberculosis (TB) surveillance,
early warning identification, and hotspot monitoring using routinely collected clinical data
in Zimbabwe.

The system is designed for low-resource healthcare settings and supports healthcare workers
through simplified surveillance-based risk assessment.

---
""")

# =========================================================
# IMPORTANT DISCLAIMER
# =========================================================

st.warning("""
DISCLAIMER:

This system is a surveillance and clinical decision-support prototype.

It DOES NOT diagnose tuberculosis.

The system is intended to:
- Support pediatric TB surveillance
- Assist early warning identification
- Monitor hotspot locations
- Support healthcare decision-making
- Improve epidemiological surveillance

Final diagnosis and treatment decisions remain the responsibility
of qualified healthcare professionals.
""")

# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

rf_model = joblib.load("randomforest.pkl")
lr_model = joblib.load("logistic.pkl")
svm_model = joblib.load("svm.pkl")
xgb_model = joblib.load("xgboost.pkl")

# =========================================================
# DATABASE FILE
# =========================================================

DATABASE = "tb_surveillance_data.csv"

# =========================================================
# CREATE DATABASE IF IT DOES NOT EXIST
# =========================================================

if not os.path.exists(DATABASE):

    empty_df = pd.DataFrame(columns=[
        "Date",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "GeneXpert",
        "Surveillance_Result"
    ])

    empty_df.to_csv(DATABASE, index=False)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATABASE)

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Patient Clinical Information")

age = st.sidebar.slider(
    "Age",
    min_value=0,
    max_value=18,
    value=5
)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"]
)

hiv_status = st.sidebar.selectbox(
    "HIV Status",
    ["Positive", "Negative", "Unknown"]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "Gweru",
        "Kwekwe",
        "Zvishavane",
        "Shurugwi",
        "Gokwe",
        "Mberengwa",
        "Mvuma"
    ]
)

xray = st.sidebar.selectbox(
    "X-ray Suggestive of TB",
    ["Yes", "No"]
)

eptb = st.sidebar.selectbox(
    "EPTB Signs Present",
    ["Yes", "No"]
)

genexpert = st.sidebar.selectbox(
    "GeneXpert Result",
    ["Positive", "Negative", "Not Available"]
)

# =========================================================
# INSTRUCTIONS
# =========================================================

st.markdown("""
## Healthcare Worker Instructions

1. Enter patient clinical information using the sidebar.
2. Click **Run Surveillance Assessment**.
3. Review the surveillance result.
4. Patient surveillance records are automatically saved.
5. Use the dashboard below to monitor trends and hotspot locations.

---
""")

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.sidebar.button("Run Surveillance Assessment"):

    # =====================================================
    # ENCODE INPUTS
    # =====================================================

    sex_val = 1 if sex == "Male" else 0
    hiv_val = 1 if hiv_status == "Positive" else 0
    xray_val = 1 if xray == "Yes" else 0
    eptb_val = 1 if eptb == "Yes" else 0

    # =====================================================
    # SIMPLE INPUT ARRAY
    # =====================================================

    input_data = np.array([[
        age,
        sex_val,
        hiv_val,
        xray_val,
        eptb_val
    ]])

    # =====================================================
    # MODEL PREDICTIONS
    # =====================================================

    rf_pred = rf_model.predict(input_data)[0]
    lr_pred = lr_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    predictions = [rf_pred, lr_pred, svm_pred, xgb_pred]

    positive_votes = predictions.count(1)

    # =====================================================
    # SURVEILLANCE LOGIC
    # =====================================================

    if genexpert == "Positive":

        final_result = "Confirmed Pediatric TB Surveillance Case"

        st.error("""
        Confirmed Pediatric TB Surveillance Case Recorded
        """)

        st.warning(f"""
        ALERT:
        {location} has recorded a confirmed pediatric TB surveillance case.
        Continue hotspot monitoring and epidemiological surveillance.
        """)

    elif positive_votes >= 3:

        final_result = "Higher Pediatric TB Surveillance Concern"

        st.warning("""
        Higher Pediatric TB Surveillance Concern Detected
        """)

        st.info("""
        Recommend further clinical investigation and laboratory testing.
        """)

    else:

        final_result = "Lower Pediatric TB Surveillance Concern"

        st.success("""
        Lower Pediatric TB Surveillance Concern
        """)

        st.info("""
        Continue monitoring and routine follow-up.
        """)

    # =====================================================
    # SAVE RECORD
    # =====================================================

    new_record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Sex": sex,
        "HIV_Status": hiv_status,
        "Location": location,
        "Xray": xray,
        "EPTB": eptb,
        "GeneXpert": genexpert,
        "Surveillance_Result": final_result
    }])

    new_record.to_csv(
        DATABASE,
        mode='a',
        header=False,
        index=False
    )

    st.success("Patient surveillance record saved successfully.")

# =========================================================
# DELETE BUTTON
# =========================================================

if st.sidebar.button("Delete All Records"):

    empty_df = pd.DataFrame(columns=[
        "Date",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "GeneXpert",
        "Surveillance_Result"
    ])

    empty_df.to_csv(DATABASE, index=False)

    st.warning("All surveillance records deleted.")

    st.rerun()

# =========================================================
# DASHBOARD
# =========================================================

st.markdown("---")

st.header("Surveillance Dashboard")

if len(df) > 0:

    # =====================================================
    # SUMMARY STATISTICS
    # =====================================================

    total_cases = len(df)

    high_cases = len(
        df[
            df["Surveillance_Result"] ==
            "Higher Pediatric TB Surveillance Concern"
        ]
    )

    confirmed_cases = len(
        df[
            df["GeneXpert"] == "Positive"
        ]
    )

    hotspot = (
        df["Location"]
        .value_counts()
        .idxmax()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_cases)
    col2.metric("High Concern Cases", high_cases)
    col3.metric("Confirmed GeneXpert Cases", confirmed_cases)
    col4.metric("Highest Hotspot", hotspot)

    st.markdown("---")

    # =====================================================
    # HOTSPOT CHART
    # =====================================================

    st.subheader("TB Hotspot Surveillance")

    hotspot_df = (
        df["Location"]
        .value_counts()
        .reset_index()
    )

    hotspot_df.columns = ["Location", "Cases"]

    fig = px.bar(
        hotspot_df,
        x="Location",
        y="Cases",
        title="Pediatric TB Surveillance Cases by Location"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SURVEILLANCE TREND
    # =====================================================

    st.subheader("Surveillance Trend")

    trend_df = (
        df["Surveillance_Result"]
        .value_counts()
        .reset_index()
    )

    trend_df.columns = ["Result", "Count"]

    fig2 = px.pie(
        trend_df,
        names="Result",
        values="Count",
        title="Distribution of Surveillance Outcomes"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # DATA TABLE
    # =====================================================

    st.subheader("Recorded Surveillance Data")

    st.dataframe(df)

else:

    st.info("No surveillance records available yet.")