# breast_model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Datasets/breast_cancer.csv")

# Select only useful features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

X = df[features]
y = df['diagnosis']  # M = Malignant, B = Benign

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/breast_cancer_model.pkl")
print("✅ Breast cancer model saved.")
