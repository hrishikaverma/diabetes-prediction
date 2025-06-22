import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from fpdf import FPDF
import csv

# ---------- Initialize user history file ----------
history_file = "user_history.csv"
columns = [
    "Timestamp", "Name", "Email", "Address", "BloodGroup",
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Prediction"
]

if not os.path.exists(history_file):
    pd.DataFrame(columns=columns).to_csv(history_file, index=False, quoting=csv.QUOTE_ALL)

# ---------- Page Config ----------
st.set_page_config(page_title="GlucoPredict – Early Diabetes Alert", layout="centered")

# ---------- Background Style ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                        url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #fff !important;
            text-shadow: 1px 1px 3px black;
        }}
        .subtitle, p, label, .stMarkdown, .stNumberInput label {{
            color: #f0f0f0 !important;
        }}
        input, .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.1);
            color: #fff !important;
        }}
        .stButton>button, .stDownloadButton>button {{
            background-color: #2e7d32;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg("Bg.png")

# ---------- Load Model and Data ----------
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    data = pd.read_csv('diabetes.csv')
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ---------- ROC Curve Function ----------
def plot_roc_curve(model, X_test, y_test):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return fig
    except Exception as e:
        st.warning(f"ROC Error: {e}")
        return None

# ---------- Sidebar ----------
st.sidebar.title("📊 Data Insights")

if st.sidebar.checkbox("Show Data Head"):
    st.sidebar.dataframe(data.head())

if st.sidebar.checkbox("Class Distribution"):
    st.sidebar.bar_chart(data['Outcome'].value_counts())

if st.sidebar.checkbox("Show Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.sidebar.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.title("🧪 Model Evaluation")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_scaled = scaler.transform(X)

if st.sidebar.checkbox("📌 Show Evaluation Metrics"):
    y_pred = model.predict(X_scaled)
    st.sidebar.subheader("🔹 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.sidebar.pyplot(fig_cm)

    st.sidebar.subheader("🔹 Classification Report")
    report_text = classification_report(y, y_pred, output_dict=False)
    st.sidebar.code(report_text)

    st.sidebar.subheader("🔹 ROC Curve")
    fig_roc = plot_roc_curve(model, X_scaled, y)
    if fig_roc:
        st.sidebar.pyplot(fig_roc)

# ---------- Main Title ----------
st.title("🯪 GlucoPredict – Early Diabetes Alert")
st.markdown('<p class="subtitle">Use manual input or upload a CSV to get predictions.</p>', unsafe_allow_html=True)

tabs = st.tabs(["📝 Manual Input", "📂 CSV Upload", "📜 User History"])
tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]

# ---------- Tab 1: Manual Input ----------
with tab1:
    st.subheader("🧾 Enter your personal & medical details:")

    with st.form("input_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        address = st.text_area("Address")
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose", 1, 200, 120)
            bp = st.number_input("Blood Pressure", 1, 140, 70)
            skin = st.number_input("Skin Thickness", 1, 100, 20)
        with col2:
            insulin = st.number_input("Insulin", 1, 900, 80)
            bmi = st.number_input("BMI", 1.0, 70.0, 25.0, format="%.2f")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, format="%.3f")
            age = st.number_input("Age", 1, 120, 30)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        if not name or not email or not address:
            st.warning("⚠️ Please fill out all personal details (Name, Email, Address) before submitting.")
        else:
            try:
                input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0]

                st.subheader("📢 Prediction Result:")
                if prediction == 1:
                    st.error("⚠️ You may have **diabetes**. Please consult a doctor.")
                else:
                    st.success("✅ You are likely not diabetic. Keep it up! 💪")

                new_row = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name,
                    "Email": email,
                    "Address": address,
                    "BloodGroup": blood_group,
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                    "Prediction": "Diabetic" if prediction == 1 else "Not Diabetic"
                }])

                if not os.path.exists(history_file) or os.stat(history_file).st_size == 0:
                    new_row.to_csv(history_file, index=False, quoting=csv.QUOTE_ALL)
                else:
                    new_row.to_csv(history_file, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)

                st.success("✅ User data and prediction saved to history.")
                st.dataframe(new_row)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# ---------- Tab 2: CSV Upload ----------
with tab2:
    st.subheader("📂 Upload CSV File")
    st.markdown("Your CSV must have the following columns:")
    st.code("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:", df.head())

            required_cols = columns[5:-1]
            if not all(col in df.columns for col in required_cols):
                missing_cols = list(set(required_cols) - set(df.columns))
                st.error(f"Missing columns: {missing_cols}")
            else:
                scaled_input = scaler.transform(df[required_cols])
                predictions = model.predict(scaled_input)
                df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
                st.success("✅ Prediction done.")
                st.dataframe(df)

                csv_result = df.to_csv(index=False).encode()
                st.download_button("📅 Download Results", csv_result, "prediction_results.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ---------- Tab 3: User History ----------
with tab3:
    st.subheader("📜 User Prediction History")
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file, on_bad_lines='skip')
            history_df = history_df.dropna(axis=1, how='all')
            if history_df.empty:
                st.info("No prediction history found yet.")
            else:
                st.dataframe(history_df)

                st.download_button(
                    "⬇ Download History CSV",
                    history_df.to_csv(index=False).encode(),
                    "user_prediction_history.csv",
                    "text/csv"
                )

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "User Prediction History", 0, 1, 'C')
                pdf.set_font("Arial", size=10)
                pdf.ln(4)

                page_width = pdf.w - 2 * pdf.l_margin
                col_width = page_width / len(history_df.columns)

                for col_name in history_df.columns:
                    pdf.cell(col_width, 10, col_name, border=1)
                pdf.ln()

                for i, row in history_df.iterrows():
                    for item in row:
                        text = str(item)
                        if len(text) > 15:
                            text = text[:12] + "..."
                        pdf.cell(col_width, 10, text, border=1)
                    pdf.ln()
                    if i >= 49:
                        pdf.cell(0, 10, "... more rows omitted ...", 0, 1, 'C')
                        break

                st.download_button(
                    "⬇ Download History PDF",
                    pdf.output(dest='S').encode('latin1'),
                    "user_prediction_history.pdf",
                    "application/pdf"
                )
        except Exception as e:
            st.error(f"Error reading history file: {e}")
    else:
        st.warning("No history file found yet.")
