import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Childhood TB Prediction System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("Machine Learning-Based Childhood TB Prediction and Surveillance System")

st.info("""
Prototype Notice:
This system is a prototype developed for research and educational purposes.
It supports preliminary childhood TB risk prediction and surveillance monitoring.
The system does not replace professional medical diagnosis.
""")

# =========================================================
# USER INSTRUCTIONS
# =========================================================

st.markdown("## Instructions for Healthcare Workers")

st.markdown("""
1. Enter patient clinical information using the left sidebar.
2. Click **Predict TB Risk**.
3. Review the prediction results.
4. Patient surveillance records are automatically saved.
5. Click **View Surveillance Dashboard** to analyse trends and hotspot locations.
""")

# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================

rf_model = joblib.load("randomforest.pkl")

lr_model = joblib.load("logistic.pkl")

svm_model = joblib.load("svm.pkl")

xgb_model = joblib.load("xgboost.pkl")

model_columns = joblib.load("model_columns.pkl")

# =========================================================
# DATABASE
# =========================================================

DATABASE = "tb_surveillance_data.csv"

if not os.path.exists(DATABASE):

    df = pd.DataFrame(columns=[
        "Date",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "TB_Outcome"
    ])

    df.to_csv(DATABASE, index=False)

# =========================================================
# LOAD DATABASE
# =========================================================

try:

    df = pd.read_csv(DATABASE)

except:

    df = pd.DataFrame(columns=[
        "Date",
        "Age",
        "Sex",
        "HIV_Status",
        "Location",
        "Xray",
        "EPTB",
        "TB_Outcome"
    ])

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Patient Clinical Information")

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=5
)

sex = st.sidebar.selectbox(
    "Sex",
    [
        "Male",
        "Female"
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
        "Gokwe",
        "Gweru",
        "Kwekwe",
        "Mberengwa",
        "Shurugwi",
        "Zvishavane",
        "Mvuma"
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
    "EPTB",
    [
        "Unknown",
        "Present"
    ]
)

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=model_columns
)

# =========================================================
# FILL INPUT DATA
# =========================================================

input_data["Age"] = age

# =========================================================
# SEX
# =========================================================

if sex == "Male":

    input_data["Sex_m"] = 1

    if "Sex_m " in input_data.columns:

        input_data["Sex_m "] = 1

# =========================================================
# HIV STATUS
# =========================================================

if hiv_status == "Positive":

    input_data["HIV_Status_1"] = 1

elif hiv_status == "Unknown":

    if "HIV_Status_unknown" in input_data.columns:

        input_data["HIV_Status_unknown"] = 1

    if "HIV_Status_uknown" in input_data.columns:

        input_data["HIV_Status_uknown"] = 1

else:

    input_data["HIV_Status_9"] = 1

# =========================================================
# LOCATION
# =========================================================

location_column = f"Location_{location}"

if location_column in input_data.columns:

    input_data[location_column] = 1

if location == "Gweru":

    if "Location_Gweru " in input_data.columns:

        input_data["Location_Gweru "] = 1

# =========================================================
# XRAY
# =========================================================

if xray == "Yes":

    input_data["X-ray_1"] = 1

elif xray == "Unknown":

    input_data["X-ray_unknown"] = 1

# =========================================================
# EPTB
# =========================================================

if eptb == "Unknown":

    input_data["EPTB_unknown"] = 1

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.sidebar.button("Predict TB Risk"):

    rf_pred = rf_model.predict(input_data)[0]

    lr_pred = lr_model.predict(input_data)[0]

    svm_pred = svm_model.predict(input_data)[0]

    xgb_pred = xgb_model.predict(input_data)[0]

    # =====================================================
    # RESULTS
    # =====================================================

    st.header("Prediction Results")

    results = pd.DataFrame({

        "Model": [
            "Random Forest",
            "Logistic Regression",
            "SVM",
            "XGBoost"
        ],

        "Prediction": [
            rf_pred,
            lr_pred,
            svm_pred,
            xgb_pred
        ]
    })

    st.dataframe(results)

    # =====================================================
    # MAJORITY VOTING
    # =====================================================

    positive_votes = (
        rf_pred +
        lr_pred +
        svm_pred +
        xgb_pred
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    if positive_votes >= 2:

        final_prediction = "Positive"

        st.error(
            "HIGH TB RISK DETECTED"
        )

    else:

        final_prediction = "Negative"

        st.success(
            "LOW TB RISK"
        )

    # =====================================================
    # SAVE SURVEILLANCE RECORD
    # =====================================================

    new_record = pd.DataFrame([{

        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),

        "Age": age,

        "Sex": sex,

        "HIV_Status": hiv_status,

        "Location": location,

        "Xray": xray,

        "EPTB": eptb,

        "TB_Outcome": final_prediction
    }])

    new_record.to_csv(
        DATABASE,
        mode='a',
        header=False,
        index=False
    )

    st.warning(
        "Patient surveillance record saved successfully."
    )

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
        "TB_Outcome"
    ])

    empty_df.to_csv(DATABASE, index=False)

    st.warning("All surveillance records deleted.")

    st.rerun()

# =========================================================
# DASHBOARD BUTTON
# =========================================================

show_dashboard = st.button("View Surveillance Dashboard")

# =========================================================
# DASHBOARD
# =========================================================

if show_dashboard:

    st.header("TB Surveillance Dashboard")

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    total_cases = len(df)

    positive_cases = len(
        df[df["TB_Outcome"] == "Positive"]
    )

    negative_cases = len(
        df[df["TB_Outcome"] == "Negative"]
    )

    col1.metric(
        "Total Cases",
        total_cases
    )

    col2.metric(
        "Positive Cases",
        positive_cases
    )

    col3.metric(
        "Negative Cases",
        negative_cases
    )

    # =====================================================
    # HOTSPOT ANALYSIS
    # =====================================================

    st.subheader("TB Hotspot Analysis")

    hotspot = (
        df[df["TB_Outcome"] == "Positive"]
        .groupby("Location")
        .size()
        .reset_index(name="Positive Cases")
    )

    if len(hotspot) > 0:

        hotspot = hotspot.sort_values(
            by="Positive Cases",
            ascending=False
        )

        fig_hotspot = px.bar(
            hotspot,
            x="Location",
            y="Positive Cases",
            text="Positive Cases",
            title="Positive TB Cases by Location"
        )

        st.plotly_chart(
            fig_hotspot,
            use_container_width=True
        )

        highest = hotspot.iloc[0]

        st.error(
            f"ALERT: {highest['Location']} currently has "
            f"the highest number of positive TB cases."
        )

    # =====================================================
    # HIV DISTRIBUTION
    # =====================================================

    st.subheader("HIV Status Distribution")

    hiv_counts = (
        df["HIV_Status"]
        .value_counts()
        .reset_index()
    )

    hiv_counts.columns = [
        "HIV Status",
        "Count"
    ]

    fig_hiv = px.pie(
        hiv_counts,
        names="HIV Status",
        values="Count",
        title="HIV Status Distribution"
    )

    st.plotly_chart(
        fig_hiv,
        use_container_width=True
    )

    # =====================================================
    # XRAY DISTRIBUTION
    # =====================================================

    st.subheader("X-Ray Distribution")

    fig_xray = px.histogram(
        df,
        x="Xray",
        color="TB_Outcome",
        barmode="group",
        title="X-Ray Findings and TB Outcome"
    )

    st.plotly_chart(
        fig_xray,
        use_container_width=True
    )

    # =====================================================
    # EPTB DISTRIBUTION
    # =====================================================

    st.subheader("EPTB Distribution")

    fig_eptb = px.histogram(
        df,
        x="EPTB",
        color="TB_Outcome",
        barmode="group",
        title="EPTB Distribution"
    )

    st.plotly_chart(
        fig_eptb,
        use_container_width=True
    )

    # =====================================================
    # RECORDED DATA
    # =====================================================

    st.subheader("Recorded Surveillance Data")

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
Developed for Childhood Tuberculosis Prediction,
Clinical Decision Support,
and Epidemiological Surveillance.
""")