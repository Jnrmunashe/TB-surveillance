import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TB Surveillance System",
    layout="wide"
)

st.title("TB Surveillance and Clinical Decision Support System")

st.info("""
Prototype Notice:
This TB Surveillance and Clinical Decision Support System is a prototype developed for research and educational purposes.
The system is designed to support TB hotspot surveillance, epidemiological monitoring, and preliminary clinical decision-making.
It should not replace professional medical diagnosis or national TB program guidelines.
""")

st.markdown("""
This system assists healthcare workers in:
- Capturing pediatric TB surveillance information
- Monitoring TB hotspot locations
- Supporting clinical decision making
- Tracking epidemiological trends
""")

# =========================================================
# DATABASE FILE
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

df = pd.read_csv(DATABASE)

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
    ["Male", "Female", "Unknown"]
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
        "Mvuma",
        "Bulawayo",
        "Harare",
        "Unknown"
    ]
)

xray = st.sidebar.selectbox(
    "X-Ray Suggestive of TB",
    ["Yes", "No", "Unknown"]
)

eptb = st.sidebar.selectbox(
    "EPTB Present",
    ["Yes", "No", "Unknown"]
)

tb_outcome = st.sidebar.selectbox(
    "TB Outcome",
    ["Positive", "Negative", "Unknown"]
)

# =========================================================
# SAVE BUTTON
# =========================================================

if st.sidebar.button("Save Surveillance Record"):

    new_record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Age": age,
        "Sex": sex,
        "HIV_Status": hiv_status,
        "Location": location,
        "Xray": xray,
        "EPTB": eptb,
        "TB_Outcome": tb_outcome
    }])

    new_record.to_csv(
        DATABASE,
        mode='a',
        header=False,
        index=False
    )

    st.success("Record saved successfully.")

    st.rerun()

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

    st.warning("All records deleted.")

    st.rerun()

# =========================================================
# DASHBOARD METRICS
# =========================================================

st.header("Overall Surveillance Statistics")

col1, col2, col3, col4 = st.columns(4)

total_cases = len(df)

positive_cases = len(df[df["TB_Outcome"] == "Positive"])

negative_cases = len(df[df["TB_Outcome"] == "Negative"])

hiv_positive = len(df[df["HIV_Status"] == "Positive"])

col1.metric("Total Records", total_cases)
col2.metric("TB Positive", positive_cases)
col3.metric("TB Negative", negative_cases)
col4.metric("HIV Positive", hiv_positive)

# =========================================================
# HOTSPOT ANALYSIS
# =========================================================

st.header("TB Hotspot Surveillance")

if len(df) > 0:

    hotspot = (
        df[df["TB_Outcome"] == "Positive"]
        .groupby("Location")
        .size()
        .reset_index(name="Positive Cases")
    )

    if len(hotspot) > 0:

        fig_hotspot = px.bar(
            hotspot,
            x="Location",
            y="Positive Cases",
            title="TB Positive Cases by Location",
            text_auto=True
        )

        st.plotly_chart(fig_hotspot, use_container_width=True)

        highest = hotspot.sort_values(
            by="Positive Cases",
            ascending=False
        ).iloc[0]

        st.error(
            f"ALERT: {highest['Location']} currently has "
            f"the highest TB burden with "
            f"{highest['Positive Cases']} positive cases."
        )

# =========================================================
# HIV STATUS DISTRIBUTION
# =========================================================

st.header("HIV Status Distribution")

if len(df) > 0:

    hiv_fig = px.pie(
        df,
        names="HIV_Status",
        title="HIV Status Distribution"
    )

    st.plotly_chart(hiv_fig, use_container_width=True)

# =========================================================
# X-RAY DISTRIBUTION
# =========================================================

st.header("X-Ray TB Suggestion Distribution")

if len(df) > 0:

    xray_fig = px.histogram(
        df,
        x="Xray",
        color="TB_Outcome",
        barmode="group",
        title="X-Ray Suggestive of TB"
    )

    st.plotly_chart(xray_fig, use_container_width=True)

# =========================================================
# EPTB DISTRIBUTION
# =========================================================

st.header("EPTB Distribution")

if len(df) > 0:

    eptb_fig = px.histogram(
        df,
        x="EPTB",
        color="TB_Outcome",
        barmode="group",
        title="EPTB Distribution"
    )

    st.plotly_chart(eptb_fig, use_container_width=True)

# =========================================================
# SURVEILLANCE TABLE
# =========================================================

st.header("Recorded Surveillance Data")

st.dataframe(df, use_container_width=True)

# =========================================================
# DOWNLOAD DATASET
# =========================================================

csv = df.to_csv(index=False)

st.download_button(
    label="Download Surveillance Dataset",
    data=csv,
    file_name="tb_surveillance_data.csv",
    mime="text/csv"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
Developed for Pediatric Tuberculosis Surveillance,
Clinical Decision Support,
and Epidemiological Monitoring.
""")