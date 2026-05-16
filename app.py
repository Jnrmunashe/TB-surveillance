import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import plotly.express as px
import time

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Pediatric TB Surveillance System",
    layout="wide"
)

# =========================================================
# DATABASE CONNECTION
# =========================================================

conn = sqlite3.connect(
    "tb_surveillance.db",
    check_same_thread=False
)

cursor = conn.cursor()

# =========================================================
# CREATE DATABASE TABLE
# =========================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS surveillance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Date TEXT,
    Age INTEGER,
    Sex TEXT,
    HIV_Status TEXT,
    Location TEXT,
    Xray TEXT,
    EPTB TEXT,
    GeneXpert TEXT,
    Surveillance_Result TEXT
)
""")

conn.commit()

# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

rf_model = joblib.load("randomforest.pkl")
lr_model = joblib.load("logistic.pkl")
svm_model = joblib.load("svm.pkl")
xgb_model = joblib.load("xgboost.pkl")

# =========================================================
# LOAD MODEL COLUMNS
# =========================================================

model_columns = joblib.load("model_columns.pkl")

# =========================================================
# TITLE
# =========================================================

st.title(
    "Machine Learning-Supported Pediatric TB Surveillance System"
)

st.markdown("""
Prototype surveillance and clinical decision-support system
for monitoring pediatric tuberculosis trends and hotspot
locations using routinely collected clinical data.
""")

# =========================================================
# DISCLAIMER
# =========================================================

st.warning("""
Prototype System:
This platform supports pediatric TB surveillance
and clinical decision-support.

It DOES NOT replace professional medical diagnosis,
laboratory confirmation, GeneXpert testing,
or clinician judgement.
""")

# =========================================================
# HEALTHCARE WORKER INSTRUCTIONS
# =========================================================

with st.expander("Healthcare Worker Instructions"):

    st.markdown("""
1. Enter patient clinical information using the sidebar.

2. Click **Run Surveillance Assessment**.

3. Review the surveillance outcome.

4. Patient records are automatically saved into the database.

5. Use the dashboard to monitor hotspot locations and surveillance trends.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    step=1
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
    "Location / District",
    [
        "Bulawayo",
        "Gokwe",
        "Gweru",
        "Kwekwe",
        "Mberengwa",
        "Mvuma",
        "Shurugwi",
        "Zvishavane",
        "Unknown"
    ]
)

xray = st.sidebar.selectbox(
    "X-ray Suggestive of TB",
    ["Yes", "No", "Unknown"]
)

eptb = st.sidebar.selectbox(
    "EPTB Signs Present",
    ["Yes", "No", "Unknown"]
)

genexpert = st.sidebar.selectbox(
    "GeneXpert Result",
    ["Positive", "Negative", "Not Available"]
)

# =========================================================
# CLEAR DISPLAY BUTTON
# =========================================================

if st.sidebar.button("Clear Display"):

    st.session_state.clear()
    st.rerun()

# =========================================================
# CLEAR DATASET BUTTON
# =========================================================

if st.sidebar.button("Clear Dataset"):

    cursor.execute("DELETE FROM surveillance_records")
    conn.commit()

    st.sidebar.success(
        "Dataset cleared successfully."
    )

# =========================================================
# RUN BUTTON
# =========================================================

run_button = st.sidebar.button(
    "Run Surveillance Assessment"
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_input():

    data = {
        'Age': age,

        'Sex_m': 1 if sex == "Male" else 0,
        'Sex_m ': 0,

        'HIV_Status_1': 1 if hiv_status == "Positive" else 0,
        'HIV_Status_9': 1 if hiv_status == "Negative" else 0,
        'HIV_Status_uknown': 1 if hiv_status == "Unknown" else 0,
        'HIV_Status_unknown': 0,

        'Location_Gokwe': 1 if location == "Gokwe" else 0,
        'Location_Gweru': 1 if location == "Gweru" else 0,
        'Location_Gweru ': 0,
        'Location_Kwekwe': 1 if location == "Kwekwe" else 0,
        'Location_Mberengwa': 1 if location == "Mberengwa" else 0,
        'Location_Shurugwi': 1 if location == "Shurugwi" else 0,
        'Location_Zvishavane': 1 if location == "Zvishavane" else 0,
        'Location_mvuma': 1 if location == "Mvuma" else 0,

        'X-ray_1': 1 if xray == "Yes" else 0,
        'X-ray_unknown': 1 if xray == "Unknown" else 0,

        'EPTB_unknown': 1 if eptb == "Unknown" else 0
    }

    input_df = pd.DataFrame([data])

    # =====================================================
    # MATCH TRAINING COLUMNS
    # =====================================================

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    return input_df

# =========================================================
# RUN SURVEILLANCE
# =========================================================

if run_button:

    input_data = prepare_input()

    rf_pred = rf_model.predict(input_data)[0]
    lr_pred = lr_model.predict(input_data)[0]
    svm_pred = svm_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    predictions = [
        rf_pred,
        lr_pred,
        svm_pred,
        xgb_pred
    ]

    final_prediction = int(
        round(np.mean(predictions))
    )

    st.subheader("Surveillance Result")

    # =====================================================
    # SURVEILLANCE LOGIC
    # =====================================================

    if genexpert == "Positive":

        result = "Confirmed GeneXpert Positive"

        st.error("""
⚠ Confirmed GeneXpert Positive Pediatric TB Surveillance Case
""")

    elif final_prediction == 1:

        result = "Higher Pediatric TB Surveillance Concern"

        st.error("""
⚠ Higher Pediatric TB Surveillance Concern Detected
""")

    else:

        result = "Lower Pediatric TB Surveillance Concern"

        st.success("""
Lower Pediatric TB Surveillance Concern
""")

    # =====================================================
    # SAVE TO DATABASE
    # =====================================================

    cursor.execute("""
    INSERT INTO surveillance_records (
        Date,
        Age,
        Sex,
        HIV_Status,
        Location,
        Xray,
        EPTB,
        GeneXpert,
        Surveillance_Result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(datetime.now()),
        age,
        sex,
        hiv_status,
        location,
        xray,
        eptb,
        genexpert,
        result
    ))

    conn.commit()

    success_message = st.success(
        "Patient surveillance record saved successfully."
    )

    time.sleep(2)

    success_message.empty()

# =========================================================
# LOAD DATABASE
# =========================================================

df = pd.read_sql_query(
    "SELECT * FROM surveillance_records",
    conn
)

# =========================================================
# DASHBOARD
# =========================================================

if not df.empty:

    st.subheader("Surveillance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    total_cases = len(df)

    high_concern = len(
        df[
            df["Surveillance_Result"]
            == "Higher Pediatric TB Surveillance Concern"
        ]
    )

    gene_positive = len(
        df[df["GeneXpert"] == "Positive"]
    )

    hotspot = (
        df["Location"].value_counts().idxmax()
        if not df.empty else "No Data"
    )

    col1.metric("Total Cases", total_cases)
    col2.metric("High Concern", high_concern)
    col3.metric("GeneXpert +", gene_positive)
    col4.metric("Top Hotspot", hotspot)

    # =====================================================
    # HOTSPOT GRAPH
    # =====================================================

    st.subheader("TB Hotspot Surveillance")

    hotspot_data = (
        df["Location"]
        .value_counts()
        .reset_index()
    )

    hotspot_data.columns = [
        "Location",
        "Cases"
    ]

    fig = px.bar(
        hotspot_data,
        x="Location",
        y="Cases",
        text="Cases"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # GENE XPERT GRAPH
    # =====================================================

    st.subheader("GeneXpert Surveillance Distribution")

    gene_data = (
        df["GeneXpert"]
        .value_counts()
        .reset_index()
    )

    gene_data.columns = [
        "GeneXpert",
        "Cases"
    ]

    fig2 = px.pie(
        gene_data,
        names="GeneXpert",
        values="Cases"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================================
    # DATASET
    # =====================================================

    st.subheader("Recorded Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD DATASET
    # =====================================================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="tb_surveillance_dataset.csv",
        mime="text/csv"
    )