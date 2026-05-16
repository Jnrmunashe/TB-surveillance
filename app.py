import streamlit as st
import pandas as pd
import joblib
import os

from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Pediatric TB Surveillance System",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("Machine Learning-Assisted Pediatric TB Surveillance System")

st.info("""
Prototype Notice:

This system was developed for research and educational purposes.

The system is intended to support:
- pediatric TB surveillance,
- early warning monitoring,
- hotspot identification,
- and clinical decision support.

The system DOES NOT diagnose tuberculosis
and DOES NOT replace clinicians or laboratory testing.
""")

# =====================================================
# USER INSTRUCTIONS
# =====================================================

st.markdown("## Instructions for Healthcare Workers")

st.markdown("""
1. Select the case type.
2. Enter patient clinical information.
3. Run the surveillance assessment.
4. Review surveillance risk output.
5. Confirmed TB cases can be directly recorded for epidemiological surveillance.
6. Use the dashboard to monitor hotspot locations and surveillance trends.
""")

# =====================================================
# LOAD MODELS
# =====================================================

rf_model = joblib.load("randomforest.pkl")

lr_model = joblib.load("logistic.pkl")

svm_model = joblib.load("svm.pkl")

xgb_model = joblib.load("xgboost.pkl")

model_columns = joblib.load("model_columns.pkl")

# =====================================================
# DATABASE
# =====================================================

DATABASE = "tb_surveillance_data.csv"

# =====================================================
# CREATE DATABASE IF NOT EXISTS
# =====================================================

if not os.path.exists(DATABASE):

    df = pd.DataFrame(columns=[

        "Date",
        "Case_Type",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "Surveillance_Status"

    ])

    df.to_csv(DATABASE, index=False)

# =====================================================
# LOAD DATABASE
# =====================================================

df = pd.read_csv(DATABASE)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Patient Clinical Information")

# =====================================================
# CASE TYPE
# =====================================================

case_type = st.sidebar.selectbox(
    "Case Type",
    [
        "Suspected Pediatric TB Case",
        "Confirmed Pediatric TB Case"
    ]
)

# =====================================================
# PATIENT VARIABLES
# =====================================================

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=18,
    value=5
)

sex = st.sidebar.selectbox(
    "Sex",
    [
        "Male",
        "Female",
        "Unknown"
    ]
)

hiv_status = st.sidebar.selectbox(
    "HIV Status",
    [
        "Positive",
        "Negative",
        "Unknown"
    ]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "Gweru",
        "Kwekwe",
        "Gokwe",
        "Zvishavane",
        "Shurugwi",
        "Mberengwa",
        "Mvuma",
        "Unknown"
    ]
)

xray = st.sidebar.selectbox(
    "X-Ray Suggestive of TB",
    [
        "Yes",
        "No",
        "Unknown"
    ]
)

eptb = st.sidebar.selectbox(
    "EPTB Present",
    [
        "Yes",
        "No",
        "Unknown"
    ]
)

# =====================================================
# RUN SURVEILLANCE SYSTEM
# =====================================================

if st.button("Run Surveillance Assessment"):

    # =================================================
    # CONFIRMED CASE
    # =================================================

    if case_type == "Confirmed Pediatric TB Case":

        surveillance_status = "Confirmed TB Surveillance Case"

        st.error("""
CONFIRMED PEDIATRIC TB CASE RECORDED
""")

        st.info("""
This patient already has laboratory or clinical
confirmation of tuberculosis.

The case has been added directly into the
surveillance database for:
- hotspot monitoring,
- epidemiological surveillance,
- and public health tracking.
""")

    # =================================================
    # SUSPECTED CASE
    # =================================================

    else:

        input_data = pd.DataFrame([{

            "Age": age,
            "Sex": sex,
            "HIV_Status": hiv_status,
            "Location": location,
            "Xray": xray,
            "EPTB": eptb

        }])

        # =============================================
        # ENCODE INPUTS
        # =============================================

        input_encoded = pd.get_dummies(input_data)

        # ADD MISSING COLUMNS

        for col in model_columns:

            if col not in input_encoded.columns:

                input_encoded[col] = 0

        # KEEP CORRECT COLUMN ORDER

        input_encoded = input_encoded[model_columns]

        # =============================================
        # MODEL PREDICTIONS
        # =============================================

        rf_pred = rf_model.predict(input_encoded)[0]

        lr_pred = lr_model.predict(input_encoded)[0]

        svm_pred = svm_model.predict(input_encoded)[0]

        xgb_pred = xgb_model.predict(input_encoded)[0]

        # =============================================
        # DISPLAY MODEL OUTPUTS
        # =============================================

        st.subheader("Machine Learning Surveillance Outputs")

        results_df = pd.DataFrame({

            "Model": [
                "Random Forest",
                "Logistic Regression",
                "Support Vector Machine",
                "XGBoost"
            ],

            "Surveillance Output": [
                rf_pred,
                lr_pred,
                svm_pred,
                xgb_pred
            ]
        })

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # =============================================
        # MAJORITY VOTING
        # =============================================

        positive_votes = sum([

            rf_pred,
            lr_pred,
            svm_pred,
            xgb_pred

        ])

        # =============================================
        # FINAL SURVEILLANCE OUTPUT
        # =============================================

        if positive_votes >= 2:

            surveillance_status = "High Pediatric TB Surveillance Risk"

            st.error("""
HIGH PEDIATRIC TB SURVEILLANCE RISK DETECTED

This child may require:
- further clinical review,
- GeneXpert investigation,
- follow-up surveillance,
- and additional laboratory testing.
""")

        else:

            surveillance_status = "Lower Pediatric TB Surveillance Risk"

            st.success("""
LOWER PEDIATRIC TB SURVEILLANCE RISK

Current surveillance indicators suggest
lower epidemiological concern.
""")

    # =================================================
    # SAVE RECORD
    # =================================================

    new_record = pd.DataFrame([{

        "Date": datetime.now(),

        "Case_Type": case_type,

        "Age": age,

        "Sex": sex,

        "HIV_Status": hiv_status,

        "Location": location,

        "Xray": xray,

        "EPTB": eptb,

        "Surveillance_Status": surveillance_status

    }])

    new_record.to_csv(
        DATABASE,
        mode='a',
        header=False,
        index=False
    )

    st.success("""
Patient surveillance record saved successfully.
""")

# =====================================================
# DELETE RECORDS
# =====================================================

if st.sidebar.button("Delete All Records"):

    empty_df = pd.DataFrame(columns=[

        "Date",
        "Case_Type",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "Surveillance_Status"

    ])

    empty_df.to_csv(
        DATABASE,
        index=False
    )

    st.warning("""
All surveillance records deleted.
""")

    st.rerun()

# =====================================================
# DASHBOARD
# =====================================================

st.header("Pediatric TB Surveillance Dashboard")

total_cases = len(df)

high_risk_cases = len(

    df[
        df["Surveillance_Status"] ==
        "High Pediatric TB Surveillance Risk"
    ]
)

confirmed_cases = len(

    df[
        df["Case_Type"] ==
        "Confirmed Pediatric TB Case"
    ]
)

lower_risk_cases = len(

    df[
        df["Surveillance_Status"] ==
        "Lower Pediatric TB Surveillance Risk"
    ]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    total_cases
)

col2.metric(
    "High Surveillance Risk",
    high_risk_cases
)

col3.metric(
    "Confirmed TB Cases",
    confirmed_cases
)

col4.metric(
    "Lower Surveillance Risk",
    lower_risk_cases
)

# =====================================================
# HOTSPOT ANALYSIS
# =====================================================

st.header("TB Hotspot Surveillance")

if len(df) > 0:

    hotspot_df = df[

        (
            df["Surveillance_Status"] ==
            "High Pediatric TB Surveillance Risk"
        )

        |

        (
            df["Case_Type"] ==
            "Confirmed Pediatric TB Case"
        )
    ]

    if len(hotspot_df) > 0:

        hotspot_counts = hotspot_df[
            "Location"
        ].value_counts()

        st.subheader(
            "High-Risk and Confirmed TB Cases by Location"
        )

        st.bar_chart(hotspot_counts)

        highest_location = hotspot_counts.idxmax()

        highest_cases = hotspot_counts.max()

        st.error(f"""
SURVEILLANCE ALERT:

{highest_location} currently shows the highest
number of high-risk or confirmed pediatric TB cases
({highest_cases} cases).

Public health follow-up and further investigation
may be required.
""")

    else:

        st.info("""
No hotspot alerts currently detected.
""")

# =====================================================
# RECORDED DATA
# =====================================================

st.header("Recorded Pediatric TB Surveillance Data")

st.dataframe(
    df,
    use_container_width=True
)