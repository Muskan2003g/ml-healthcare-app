# 💊 Smart Health Predictor

A beautiful AI-powered healthcare web app built with **Streamlit** to predict:

- ❤️ Heart Disease
- 🎗️ Breast Cancer
- 📊 Bulk Patient Risk Dashboard

## 🚀 Features

- Elegant Pinterest-style UI with pink/white theme
- Predicts heart disease and breast cancer from user input
- Generates PDF reports for diagnosis
- Dashboard to upload CSVs and analyze bulk patient data
- Downloadable result CSV with risk summary chart

---

## 🧠 Models Used

- `heart_disease_model.pkl` — Trained using common heart health metrics
- `breast_cancer_model.pkl` — Trained on UCI Breast Cancer dataset

---

## 📂 File Structure

ML-Healthcare-Web-App/
│
├── models/
│ ├── heart_disease_model.pkl
│ ├── breast_cancer_model.pkl
│
├── ML_Healthcare.py # Main Streamlit app
├── requirements.txt # Dependencies
├── README.md # Project details


---

## 🔧 How to Run

Make sure Python is installed.

### 1. Install packages

```bash
pip install -r requirements.txt

2. Run the app

streamlit run ML_Healthcare.py

Sample CSV Format for Dashboard
You can upload CSVs like this:
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
60,1,0,140,289,0,1,172,0,0.0,2,0,2
45,0,2,130,234,0,0,160,1,1.1,1,1,3


🧠 Tech Stack
Python

Streamlit

scikit-learn

joblib

matplotlib

FPDF

✨ Future Plans
Voice input

Login & history

Real-time explainability

More diseases

📫 Contact
Made with ❤️ by MUSKAN GAUR

## ✅ Step 3: Create `requirements.txt`

If not present, create one using the following command inside your project folder:

```bash
pip freeze > requirements.txt

✅ Step 4: Push Code to GitHub
Now follow these commands:

🧩 Open Terminal or Command Prompt:
Navigate to your project folder:

cd path/to/ML-Healthcare-Web-App

git init
git add .
git commit -m "Initial commit - ML Healthcare App"

# Replace with your actual GitHub repo link
git remote add origin https://github.com/your-username/ml-healthcare-app.git

git branch -M main
git push -u origin main


✅ Step 5: Confirm Upload
Go to your GitHub repository and verify:

All files are present

README renders properly
