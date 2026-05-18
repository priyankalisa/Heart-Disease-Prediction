import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(to right, #ffdde1, #ee9ca7);
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

h1{
color:#99004d;
text-align:center;
font-size:45px;
}

h3{
color:#660033;
text-align:center;
}

.stButton>button{
background-color:#ff4b4b;
color:white;
font-size:20px;
height:3em;
width:100%;
border-radius:15px;
border:none;
}

.stButton>button:hover{
background-color:#cc0000;
color:white;
}

div[data-baseweb="select"]{
background-color:white;
border-radius:10px;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# LOAD MODEL FILES
# =========================

model = joblib.load("LR_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# =========================
# TITLE
# =========================

st.title("❤️ Heart Disease Prediction")

st.markdown(
    "<h3>Machine Learning Based Heart Disease Detection System</h3>",
    unsafe_allow_html=True
)

st.write("")

# =========================
# USER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 1, 100, 25)

    sex = st.selectbox("Sex", ["Male", "Female"])

    max_hr = st.slider("Maximum Heart Rate", 50, 250, 150)

    fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP"]
    )

with col2:

    oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0)

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Yes", "No"]
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["ST", "Normal"]
    )

st.write("")

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Heart Disease"):

    # Create input dictionary

    raw_input = {

        'ST_Slope_Up': 1 if st_slope == "Up" else 0,

        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,

        'ChestPainType_ASY': 1 if chest_pain == "ASY" else 0,

        'is_ExerciseAngina': 1 if exercise_angina == "Yes" else 0,

        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,

        'MaxHR': max_hr,

        'is_female': 1 if sex == "Female" else 0,

        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,

        'Oldpeak': oldpeak,

        'Age': age,

        'FastingBS': fasting_bs,

        'ST_Slope_Down': 1 if st_slope == "Down" else 0,

        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,

        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0
    }

    # Convert to dataframe

    input_df = pd.DataFrame([raw_input])

    # Numerical columns

    num_cols = ['MaxHR', 'Oldpeak', 'Age']

    # Scale numerical features

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Reorder columns

    input_df = input_df[expected_columns]

    # Prediction

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)

    st.write("")

    # Output result

    if prediction == 1:

        st.error("⚠️ High Risk of Heart Disease")

    else:

        st.success("✅ Low Risk of Heart Disease")

    st.write("")

    st.subheader("Prediction Probability")

    st.write(probability)