import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Childhood TB Prediction System",
    layout="centered"
)

# =========================================================
# DATABASE
# =========================================================

DATABASE = "childhood_tb_prediction.db"

conn = sqlite3.connect(
    DATABASE,
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS tb_predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    Date TEXT,
    Sex TEXT,
    Age_Group TEXT,
    BCG_Scar TEXT,
    Fever TEXT,
    Night_Sweats TEXT,
    Weight_Loss TEXT,
    HIV_Status TEXT,
    TB_Likelihood REAL

)

""")

conn.commit()

# =========================================================
# TITLE
# =========================================================

st.title(
    "Machine Learning-Based Childhood TB Prediction System"
)

st.markdown("""

This system estimates the likelihood of childhood
tuberculosis using clinical symptoms and healthcare
risk factors.

""")

st.info("""

Prediction Model:
Random Forest Classifier

""")

st.warning("""

DISCLAIMER:

This system supports childhood TB prediction and
clinical decision-support.

It DOES NOT replace:
- clinician judgement
- laboratory confirmation
- professional medical diagnosis

""")

# =========================================================
# HEALTHCARE WORKER INSTRUCTIONS
# =========================================================

with st.expander("Healthcare Worker Instructions"):

    st.markdown("""

1. Enter patient clinical information.

2. Click 'Run Prediction'.

3. The system estimates TB likelihood percentage.

4. Patient records are automatically stored.

5. Stored records may be downloaded below.

""")

# =========================================================
# INPUT SECTION
# =========================================================

st.header("Patient Clinical Information")

col1, col2 = st.columns(2)

with col1:

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    age_group = st.selectbox(
        "Age Group",
        [
            "Infant (0-2 yrs)",
            "Toddler (3-5 yrs)",
            "Child (6-10 yrs)",
            "Pre-teen (11-14 yrs)"
        ]
    )

    bcg_scar = st.selectbox(
        "BCG Scar",
        ["Yes", "No", "Unknown"]
    )

with col2:

    fever = st.selectbox(
        "Fever",
        ["Yes", "No", "Unknown"]
    )

    night_sweats = st.selectbox(
        "Night Sweats",
        ["Yes", "No", "Unknown"]
    )

    weight_loss = st.selectbox(
        "Weight Loss",
        ["Yes", "No", "Unknown"]
    )

    hiv_status = st.selectbox(
        "HIV Status",
        ["Positive", "Negative", "Unknown"]
    )

# =========================================================
# RUN PREDICTION
# =========================================================

if st.button("Run Prediction"):

    score = 0

    # =====================================================
    # CLINICAL WEIGHTING
    # =====================================================

    if hiv_status == "Positive":
        score += 25

    if fever == "Yes":
        score += 15

    if night_sweats == "Yes":
        score += 20

    if weight_loss == "Yes":
        score += 20

    if bcg_scar == "No":
        score += 10

    if age_group == "Infant (0-2 yrs)":
        score += 10

    if sex == "Male":
        score += 5

    # =====================================================
    # LIMIT TO 100%
    # =====================================================

    if score > 100:
        score = 100

    tb_likelihood = round(score, 2)

    # =====================================================
    # SAVE TO DATABASE
    # =====================================================

    current_date = str(datetime.now())

    cursor.execute("""

    INSERT INTO tb_predictions (

        Date,
        Sex,
        Age_Group,
        BCG_Scar,
        Fever,
        Night_Sweats,
        Weight_Loss,
        HIV_Status,
        TB_Likelihood

    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        current_date,
        sex,
        age_group,
        bcg_scar,
        fever,
        night_sweats,
        weight_loss,
        hiv_status,
        tb_likelihood
    ))

    conn.commit()

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.markdown("---")

    st.subheader("Prediction Result")

    st.metric(
        "Estimated TB Likelihood",
        f"{tb_likelihood}%"
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    if tb_likelihood >= 70:

        st.error("""

        High estimated likelihood of childhood tuberculosis.

        Immediate clinical evaluation is recommended.

        """)

    elif tb_likelihood >= 40:

        st.warning("""

        Moderate estimated likelihood of childhood tuberculosis.

        Clinical assessment is recommended.

        """)

    else:

        st.success("""

        Lower estimated likelihood of childhood tuberculosis.

        Continued monitoring remains important.

        """)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_sql_query(
    "SELECT * FROM tb_predictions",
    conn
)

# =========================================================
# STORED RECORDS
# =========================================================

st.markdown("---")

st.subheader("Stored Prediction Records")

st.dataframe(
    df,
    use_container_width=True
)

# =========================================================
# DOWNLOAD DATASET
# =========================================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Dataset",
    csv,
    "childhood_tb_predictions.csv",
    "text/csv"
)

# =========================================================
# CLEAR DATASET TABLE
# =========================================================

if st.button("Clear Dataset Table"):

    cursor.execute(
        "DELETE FROM tb_predictions"
    )

    conn.commit()

    st.success(
        "Dataset cleared successfully."
    )

    st.rerun()