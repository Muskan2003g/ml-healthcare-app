# ML_Healthcare.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import base64

# Set page config
st.set_page_config(page_title="Smart Health Predictor", layout="centered")

# Inject custom CSS for Pinterest-style UI and Light/Dark Mode toggle
custom_css = """
<style>
body {
    font-family: 'Segoe UI', 'Comic Sans MS', cursive, sans-serif;
    transition: background-color 0.5s ease;
}
h1, h2, h3, h4, h5, h6, p, label {
    font-family: 'Segoe UI', 'Comic Sans MS', cursive;
    font-weight: 500;
}
button[kind="primary"] {
    background-color: #ff69b4 !important;
    color: white !important;
    border-radius: 10px;
    font-size: 16px;
}
button:hover {
    background-color: #ff1493 !important;
}
div[data-testid="stMarkdownContainer"] > div {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
section[data-testid="stSlider"] div[role="slider"] {
    background: #ff69b4;
}
details {
    background-color: #ffe4e1;
    border-radius: 8px;
    padding: 10px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Light/Dark mode toggle
# mode = st.radio("Select Mode", ["🌸 Light Mode", "🖤 Dark Mode"], horizontal=True)
# st.markdown(f"<body data-theme='{ 'light' if mode == '🌸 Light Mode' else 'dark' }'>", unsafe_allow_html=True)

# Load ML models
heart_model = joblib.load("models/heart_disease_model.pkl")
cancer_model = joblib.load("models/breast_cancer_model.pkl")

# Utility function to show prediction result
def show_prediction_card(title, result, probability, severity, color, emoji):
    st.markdown(f"""
        <div style="background-color:#fff0f5; padding: 20px; border-radius: 15px; border: 2px solid {color}; box-shadow: 0 0 10px {color}; margin-top: 20px;">
            <h2 style="color:{color}; font-size: 24px;">{emoji} {title}</h2>
            <h3 style="color:black;">Result: <span style="color:{color};">{result}</span></h3>
            <p style="font-size: 18px;">Probability: <strong>{probability:.2f}%</strong></p>
            <p style="font-size: 18px;">Severity: <strong style="color:{color};">{severity}</strong></p>
        </div>
    """, unsafe_allow_html=True)

# Utility function to generate PDF report
def generate_pdf_report(title, user_data, result, probability, severity):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 105, 180)
    pdf.cell(200, 10, txt=title, ln=1, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)

    pdf.ln(10)
    for key, value in user_data.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(255, 0, 0 if severity == "High Risk" else 255 if severity == "Moderate Risk" else 0)
    pdf.cell(0, 10, f"Result: {result}", ln=1)
    pdf.cell(0, 10, f"Probability: {probability:.2f}%", ln=1)
    pdf.cell(0, 10, f"Severity: {severity}", ln=1)

    return pdf.output(dest='S').encode('latin1')

# Tabs
tabs = st.tabs(["❤️ Heart Disease", "🎗️ Breast Cancer", "📊 Dashboard"])

# ---------------- HEART DISEASE ----------------
with tabs[0]:
    st.header("Heart Disease Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Male", "Female"], index=0)
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.slider("Resting BP (trestbps)", 80, 200, 120)
        chol = st.slider("Cholesterol (chol)", 100, 600, 240)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], index=1)
    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=1)
        thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"], index=1)
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2], index=1)
        ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3], index=0)
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=2)

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    heart_input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'chol',
                                        'fbs', 'restecg', 'thalach', 'exang',
                                        'oldpeak', 'slope', 'ca', 'thal'])

    if st.button("Predict Heart Disease"):
        try:
            # Rename features to match model's training format
            feature_mapping = {
                'trestbps': 'trtbps',
                'thalach': 'thalachh',
                'exang': 'exng',
                'ca': 'caa',
                'slope': 'slp',
                'thal': 'thall'
            }
            heart_input.rename(columns=feature_mapping, inplace=True)

            heart_input = heart_input[list(heart_model.feature_names_in_)]

            result = heart_model.predict(heart_input)[0]
            proba = heart_model.predict_proba(heart_input)[0][1]

            severity = "High Risk" if proba >= 0.8 else "Moderate Risk" if proba >= 0.5 else "Low Risk"
            color = "red" if severity == "High Risk" else "orange" if severity == "Moderate Risk" else "green"
            result_text = "Heart Disease Detected" if result == 1 else "No Heart Disease"

            show_prediction_card("Heart Disease Prediction", result_text, proba * 100, severity, color, "❤️")

            with st.expander("Download Report"):
                user_data = {
                    "Age": age, "Sex": "Male" if sex else "Female", "Chest Pain Type": cp,
                    "Resting BP": trestbps, "Cholesterol": chol, "FBS > 120": "Yes" if fbs else "No",
                    "Rest ECG": restecg, "Max Heart Rate": thalach, "Exercise Angina": "Yes" if exang else "No",
                    "Oldpeak": oldpeak, "Slope": slope, "Vessels": ca, "Thalassemia": thal
                }
                pdf_data = generate_pdf_report("Heart Disease Prediction", user_data, result_text, proba * 100, severity)
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="heart_disease_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Feature mismatch or input error: {str(e)}")

# ---------------- BREAST CANCER ----------------
with tabs[1]:
    st.header("Breast Cancer Prediction")

    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.slider("Radius Mean", 6.0, 30.0, 14.0)
        texture_mean = st.slider("Texture Mean", 10.0, 40.0, 19.0)
        perimeter_mean = st.slider("Perimeter Mean", 40.0, 200.0, 90.0)
        area_mean = st.slider("Area Mean", 100.0, 2500.0, 500.0)
    with col2:
        smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.2, 0.1)
        compactness_mean = st.slider("Compactness Mean", 0.01, 0.35, 0.15)
        concavity_mean = st.slider("Concavity Mean", 0.01, 0.4, 0.1)
        symmetry_mean = st.slider("Symmetry Mean", 0.1, 0.4, 0.2)
        fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.05, 0.1, 0.06)

    cancer_input = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean,
                                  smoothness_mean, compactness_mean, concavity_mean,
                                  symmetry_mean, fractal_dimension_mean]],
                                 columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                          'area_mean', 'smoothness_mean', 'compactness_mean',
                                          'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean'])

    if st.button("Predict Breast Cancer"):
        try:
            cancer_input = cancer_input[list(cancer_model.feature_names_in_)]
            result = cancer_model.predict(cancer_input)[0]
            proba = cancer_model.predict_proba(cancer_input)[0]

            malignant_index = list(cancer_model.classes_).index("M") if "M" in cancer_model.classes_ else 1
            malignant_proba = proba[malignant_index]

            severity = "High Risk" if malignant_proba >= 0.8 else "Moderate Risk" if malignant_proba >= 0.5 else "Low Risk"
            color = "red" if severity == "High Risk" else "orange" if severity == "Moderate Risk" else "green"
            result_label = "Malignant Tumor" if result == "M" else "Benign Tumor"

            show_prediction_card("Breast Cancer Prediction", result_label, malignant_proba * 100, severity, color, "🎗️")

            with st.expander("Download Report"):
                user_data = {
                    "Radius Mean": radius_mean, "Texture Mean": texture_mean,
                    "Perimeter Mean": perimeter_mean, "Area Mean": area_mean,
                    "Smoothness Mean": smoothness_mean, "Compactness Mean": compactness_mean,
                    "Concavity Mean": concavity_mean, "Symmetry Mean": symmetry_mean,
                    "Fractal Dimension Mean": fractal_dimension_mean
                }
                pdf_data = generate_pdf_report("Breast Cancer Prediction", user_data, result_label, malignant_proba * 100, severity)
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="breast_cancer_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Feature mismatch or input error: {str(e)}")


# ---------------- DASHBOARD TAB ----------------
with tabs[2]:
    st.header("📊 Bulk Prediction & Dashboard View")
    st.write("Upload a CSV file with patient data to analyze risk in bulk.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Data Preview:")
            st.dataframe(df.head())

            # Rename if required for heart model prediction
            feature_mapping = {
                'trestbps': 'trtbps',
                'thalach': 'thalachh',
                'exang': 'exng',
                'ca': 'caa',
                'slope': 'slp',
                'thal': 'thall'
            }
            df.rename(columns=feature_mapping, inplace=True)

            expected_features = list(heart_model.feature_names_in_)
            df = df[expected_features]

            preds = heart_model.predict(df)
            probs = heart_model.predict_proba(df)[:, 1]

            df['Prediction'] = ["Heart Disease" if x == 1 else "No Disease" for x in preds]
            df['Probability'] = probs * 100
            df['Severity'] = df['Probability'].apply(lambda p: "High Risk" if p >= 80 else "Moderate Risk" if p >= 50 else "Low Risk")

            st.success("✅ Bulk Prediction Completed!")

            # Summary Chart
            st.subheader("🧠 Prediction Severity Summary")
            st.bar_chart(df['Severity'].value_counts())

            # Table of results
            st.subheader("📋 Patient Predictions")
            st.dataframe(df[['Prediction', 'Probability', 'Severity']])

            # Download
            csv_download = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV Results", csv_download, file_name="bulk_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


