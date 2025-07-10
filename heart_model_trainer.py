# heart_model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Datasets/heart.csv")

# Split features and target
X = df.drop('output', axis=1)  # adjust 'output' if target column has different name
y = df['output']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/heart_disease_model.pkl")
print("✅ Heart disease model saved.")
