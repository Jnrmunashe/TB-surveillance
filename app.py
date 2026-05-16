import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Pediatric TB Surveillance System",
    layout="wide"
)

# =========================================================
# SESSION STATES
# =========================================================

if "show_outputs" not in st.session_state:
    st.session_state.show_outputs = False

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "show_statistics" not in st.session_state:
    st.session_state.show_statistics = False

# =========================================================
# DATABASE
# =========================================================

DATABASE = "tb_surveillance.db"

conn = sqlite3.connect(
    DATABASE,
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS surveillance_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Date TEXT,
    Age INTEGER,
    Sex TEXT,
    HIV_Status TEXT,
    Location TEXT,
    Xray TEXT,
    EPTB TEXT,
    GeneXpert TEXT,
    Surveillance_Outcome TEXT
)
""")

conn.commit()

# =========================================================
# LOAD MODELS
# =========================================================

rf_model = joblib.load("randomforest.pkl")
lr_model = joblib.load("logistic.pkl")
svm_model = joblib.load("svm.pkl")
xgb_model = joblib.load("xgboost.pkl")

model_columns = joblib.load("model_columns.pkl")

# =========================================================
# TITLE
# =========================================================

st.title(
    "Machine Learning-Supported Pediatric TB Surveillance System"
)

st.markdown("""
Prototype surveillance and clinical decision-support system for monitoring pediatric tuberculosis trends and hotspot locations using routinely collected clinical data.
""")

st.warning("""
Prototype System:
This platform supports pediatric TB surveillance and clinical decision-support.

It DOES NOT replace professional medical diagnosis,
GeneXpert testing, laboratory confirmation,
or clinician judgement.
""")

# =========================================================
# HEALTHCARE WORKER INSTRUCTIONS
# =========================================================

with st.expander("Healthcare Worker Instructions"):

    st.markdown("""
1. Enter patient clinical information.

2. Click Enter Data to save surveillance records.

3. Click Run Surveillance Assessment to display surveillance analytics.

4. Use hotspot monitoring and dashboard analytics for surveillance support.

5. Click Show Detailed Statistics to display GeneXpert, X-ray and EPTB analytics.

6. Database records remain permanently stored unless manually removed by system administrators.
""")

# =========================================================
# SIDEBAR
# =========================================================

run_assessment = st.sidebar.button(
    "Run Surveillance Assessment"
)

if st.sidebar.button("Clear Displayed Results"):

    st.session_state.show_outputs = False
    st.session_state.show_statistics = False
    st.session_state.current_result = None

    st.rerun()

# =========================================================
# INPUT SECTION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    step=1,
    key="age"
)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"],
    key="sex"
)

hiv_status = st.sidebar.selectbox(
    "HIV Status",
    ["Positive", "Negative", "Unknown"],
    key="hiv"
)

location = st.sidebar.selectbox(
    "Location / District",
    [
        "Bulawayo",
        "Gweru",
        "Kwekwe",
        "Shurugwi",
        "Zvishavane",
        "Mberengwa",
        "Gokwe",
        "Mvuma"
    ],
    key="location"
)

xray = st.sidebar.selectbox(
    "X-ray Suggestive of TB",
    ["Yes", "No", "Unknown"],
    key="xray"
)

eptb = st.sidebar.selectbox(
    "EPTB Signs Present",
    ["Yes", "No", "Unknown"],
    key="eptb"
)

genexpert = st.sidebar.selectbox(
    "GeneXpert Result",
    ["Positive", "Negative", "Not Available"],
    key="genexpert"
)

# =========================================================
# ENTER DATA BUTTON
# =========================================================

enter_data = st.sidebar.button(
    "Enter Data"
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

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    return input_df

# =========================================================
# ENTER DATA LOGIC
# =========================================================

if enter_data:

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

    # =====================================================
    # FINAL SURVEILLANCE RESULT
    # =====================================================

    if genexpert == "Positive":

        final_result = (
            "Higher Pediatric TB Surveillance Concern"
        )

    else:

        if final_prediction == 1:

            final_result = (
                "Higher Pediatric TB Surveillance Concern"
            )

        else:

            final_result = (
                "Lower Pediatric TB Surveillance Concern"
            )

    # =====================================================
    # SAVE TO DATABASE
    # =====================================================

    current_date = str(datetime.now())

    cursor.execute("""
    INSERT INTO surveillance_data (
        Date,
        Age,
        Sex,
        HIV_Status,
        Location,
        Xray,
        EPTB,
        GeneXpert,
        Surveillance_Outcome
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        current_date,
        age,
        sex,
        hiv_status,
        location,
        xray,
        eptb,
        genexpert,
        final_result
    ))

    conn.commit()

    st.sidebar.success(
        "Patient surveillance data entered successfully."
    )

# =========================================================
# RUN ASSESSMENT DISPLAY
# =========================================================

if run_assessment:

    st.session_state.show_outputs = True

# =========================================================
# LOAD DATABASE
# =========================================================

df = pd.read_sql_query(
    "SELECT * FROM surveillance_data",
    conn
)

# =========================================================
# DISPLAY OUTPUTS
# =========================================================

if st.session_state.show_outputs:

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.subheader("Surveillance Dashboard")

    total_cases = len(df)

    high_concern = len(
        df[
            df["Surveillance_Outcome"]
            ==
            "Higher Pediatric TB Surveillance Concern"
        ]
    )

    if len(df) > 0:

        top_hotspot = (
            df["Location"]
            .value_counts()
            .idxmax()
        )

    else:

        top_hotspot = "No Data"

    col1, col2, col3 = st.columns([1,1,2])

    col1.metric(
        "Total Cases",
        total_cases
    )

    col2.metric(
        "High Concern",
        high_concern
    )

    col3.markdown("### Top Hotspot")
    col3.write(f"## {top_hotspot}")

    # =====================================================
    # HOTSPOT GRAPH
    # =====================================================

    if len(df) > 0:

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
            text="Cases",
            title="TB Cases by Location"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="hotspot_chart"
        )

    # =====================================================
    # SHOW DETAILED STATISTICS BUTTON
    # =====================================================

    if st.button("Show Detailed Statistics"):

        st.session_state.show_statistics = True

    # =====================================================
    # DETAILED STATISTICS SECTION
    # =====================================================

    if st.session_state.show_statistics:

        st.subheader("Detailed Clinical Statistics")

        # =================================================
        # GENEXPERT GRAPH
        # =================================================

        gene_data = (
            df["GeneXpert"]
            .value_counts()
            .reset_index()
        )

        gene_data.columns = [
            "Result",
            "Count"
        ]

        fig2 = px.pie(
            gene_data,
            names="Result",
            values="Count",
            title="GeneXpert Results Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            key="gene_xpert_chart"
        )

        # =================================================
        # XRAY GRAPH
        # =================================================

        xray_data = (
            df["Xray"]
            .value_counts()
            .reset_index()
        )

        xray_data.columns = [
            "Result",
            "Count"
        ]

        fig3 = px.bar(
            xray_data,
            x="Result",
            y="Count",
            text="Count",
            title="X-ray Suggestive of TB"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True,
            key="xray_chart"
        )

        # =================================================
        # EPTB GRAPH
        # =================================================

        eptb_data = (
            df["EPTB"]
            .value_counts()
            .reset_index()
        )

        eptb_data.columns = [
            "Result",
            "Count"
        ]

        fig4 = px.bar(
            eptb_data,
            x="Result",
            y="Count",
            text="Count",
            title="EPTB Signs Distribution"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
            key="eptb_chart"
        )

    # =====================================================
    # DATASET TABLE
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
        "Download Dataset",
        csv,
        "tb_surveillance_dataset.csv",
        "text/csv"
    )

    # =====================================================
    # CLEAR DATASET TABLE ONLY
    # =====================================================

    if st.button("Clear Dataset Table"):

        st.session_state.show_outputs = False
        st.session_state.show_statistics = False

        st.rerun()