import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

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

# Step 4: Train All Models
logistic_model = LogisticRegression()
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

logistic_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Step 5: Evaluate Each Model
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

print("\n🎯 Model Evaluation Results:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n📌 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save All Models & Scaler
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 All models and scaler saved successfully!")
