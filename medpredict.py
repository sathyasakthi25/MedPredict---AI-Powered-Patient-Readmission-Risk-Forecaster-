# medpredict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# -------------------------------
# 1. Simulate Patient Dataset
# -------------------------------
data = pd.DataFrame({
    'age': np.random.randint(20, 90, 1000),
    'num_prev_admissions': np.random.randint(0, 5, 1000),
    'length_of_stay': np.random.randint(1, 15, 1000),
    'comorbidity_score': np.random.randint(0, 10, 1000),
    'has_diabetes': np.random.randint(0, 2, 1000),
    'has_hypertension': np.random.randint(0, 2, 1000),
    'readmitted': np.random.randint(0, 2, 1000)  # Target variable
})

# -------------------------------
# 2. Preprocess and Split
# -------------------------------
X = data.drop('readmitted', axis=1)
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Train Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 5. Explain Predictions with SHAP
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(shap_values[1], X_test)

# -------------------------------
# 6. Predict New Patient Case
# -------------------------------
new_patient = pd.DataFrame({
    'age': [68],
    'num_prev_admissions': [2],
    'length_of_stay': [6],
    'comorbidity_score': [5],
    'has_diabetes': [1],
    'has_hypertension': [0]
})

prediction = model.predict(new_patient)
risk_score = model.predict_proba(new_patient)[0][1]

print("\nNew Patient Prediction:")
print("Readmission Risk (0=Low, 1=High):", prediction[0])
print("Risk Score:", round(risk_score, 2))