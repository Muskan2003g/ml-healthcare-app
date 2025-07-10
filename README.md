# ml-healthcare-app
AI-Powered Health Predictor using Streamlit

🩺 ML Healthcare Web App
An interactive AI-powered medical diagnosis web application that predicts the likelihood of Heart Disease and Breast Cancer using machine learning models.

Built with Streamlit, styled with a clean Pinterest-inspired UI, and supports PDF report downloads.

🔍 Features
🎯 Predicts Heart Disease and Breast Cancer using trained ML models.

📋 Clean and user-friendly layout.

💡 Real-time prediction results with risk severity color-coding.

📄 Option to download PDF reports of results.

💾 Models trained using Scikit-learn, saved with joblib.

📁 Project Structure
ML-Healthcare-Web-App/
├── ML_Healthcare.py              # Main Streamlit app
├── models/
│   ├── heart_disease_model.pkl   # Heart disease ML model
│   └── breast_cancer_model.pkl   # Breast cancer ML model
├── Datasets/
│   ├── heart.csv                 # Dataset for heart disease
│   └── breast_cancer.csv         # Dataset for breast cancer
├── Results/
│   ├── r1.png ... r5.png         # UI screenshots
├── heart_model_trainer.py       # Model training script (heart)
├── breast_model_trainer.py      # Model training script (cancer)
└── README.md                     # Project documentation

🛠️ Installation & Run Locally
📦 Requirements
Python 3.8+

pip

Git
# 1. Clone the repository
git clone https://github.com/Muskan2003g/ml-healthcare-app.git
cd ml-healthcare-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run ML_Healthcare.py

💡 Make sure models/ contains both .pkl model files.

🔎 How It Works
Collects user input through sliders and dropdowns.

Preprocesses the inputs into a DataFrame.

Feeds it to the pre-trained ML model.

Displays result, probability, severity.

Allows PDF report download.

 Machine Learning Details
Heart Disease Model: Trained using logistic regression / decision trees.

Breast Cancer Model: Based on the popular UCI Breast Cancer dataset.

Feature engineering and cleaning handled in training scripts.

📄 License
This project is open-source and free to use for educational and non-commercial purposes.

🙋‍♀️ Author
Muskan Gaur
🔗 GitHub: @Muskan2003g


