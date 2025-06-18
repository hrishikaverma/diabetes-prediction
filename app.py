import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # 🆕 Added for saving the model and scaler

# Step 1: Load Dataset
data = pd.read_csv('diabetes.csv')
print("🔍 First 5 Rows of Dataset:")
print(data.head())

# Step 2: Replace 0s with NaN in specific columns
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols] = data[cols].replace(0, np.nan)

# Fill NaN with column means
for col in cols:
    data[col] = data[col].fillna(data[col].mean())

print("\nMissing values after filling:")
print(data.isnull().sum())

# Step 3: Split features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n✅ Data splitting and scaling completed.")

# Step 4: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Prediction & Accuracy Evaluation
y_pred = model.predict(X_test)

print("\n🎯 Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save Model & Scaler using joblib
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 Model and Scaler saved successfully!")
