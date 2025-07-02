
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
import requests 
import smtplib
from email.message import EmailMessage
import random
import time
import re

# ✅ Page Config
st.set_page_config(page_title="GlucoPredict – Early Diabetes Alert", layout="centered")

# ---------------------- Background Styling ----------------------
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

# ---------------------- Utility: Validate Email/Mobile ----------------------
def is_valid_email_or_mobile(user_input):
    email_regex = r'^\S+@\S+\.\S+$'
    mobile_regex = r'^\d{10}$'
    return re.match(email_regex, user_input) or re.match(mobile_regex, user_input)

# ---------------------- OTP Handling ----------------------
def send_otp():
    otp = random.randint(100000, 999999)
    st.session_state['generated_otp'] = str(otp)
    st.session_state['otp_sent_time'] = time.time()
    st.info(f"📩 Your OTP is: {otp}")  # For demo

# ---------------------- Login/Register ----------------------
def login_page():
    st.markdown("## 🔐 Login / Register with OTP")
    name = st.text_input("👤 Full Name")
    user_input = st.text_input("📧 Email / Mobile Number")

    if st.button("📨 Send OTP"):
        if not name.strip() or not user_input.strip():
            st.warning("⚠️ Please fill all fields.")
        elif not is_valid_email_or_mobile(user_input):
            st.error("❌ Invalid format.")
        else:
            st.session_state['user'] = user_input.strip()
            st.session_state['name'] = name.strip()
            send_otp()
            st.session_state['otp_sent'] = True

    if st.session_state.get('otp_sent', False):
        elapsed = int(time.time() - st.session_state.get('otp_sent_time', 0))
        time_left = max(0, 30 - elapsed)
        st.markdown("### 🔐 Enter 6-digit OTP:")
        cols = st.columns(6)
        otp_input = "".join(cols[i].text_input(f"{i+1}", max_chars=1, key=f"otp{i}") for i in range(6))

        if st.button("✅ Verify OTP"):
            if otp_input == st.session_state.get('generated_otp'):
                st.success(f"🎉 Welcome, {st.session_state['name']}!")
                st.balloons()
                st.session_state['logged_in'] = True

                df = pd.read_csv("user_history.csv") if os.path.exists("user_history.csv") else pd.DataFrame(columns=["Name", "Email", "Timestamp"])
                if st.session_state['user'] not in df['Email'].values:
                    df.loc[len(df)] = [st.session_state['name'], st.session_state['user'], datetime.now()]
                    df.to_csv("user_history.csv", index=False)
                st.rerun()
            else:
                st.error("❌ Invalid OTP.")

        if time_left > 0:
            st.info(f"⏳ Resend OTP in {time_left}s")
        else:
            if st.button("🔁 Resend OTP"):
                send_otp()

# ---------------------- Logout ----------------------
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------------- Main App ----------------------
def main_app():
    st.title("🩺 GlucoPredict Dashboard")
    st.success(f"👋 Welcome back, {st.session_state['name']}!")
    if st.button("Logout 🔒"):
        logout()

# ---------------------- Routing ----------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    set_bg("Bg.jpg")
    login_page()
    st.stop()
else:
    set_bg("Bg.png")
    main_app()




# ---------------------- Background Style ----------------------
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

# ---------------------- Initialize History File ----------------------
history_file = "user_history.csv"
columns = [
    "Timestamp", "Name", "Email", "Address", "BloodGroup",
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Prediction"
]

if not os.path.exists(history_file):
    pd.DataFrame(columns=columns).to_csv(history_file, index=False, quoting=csv.QUOTE_ALL)

# ---------------------- Load Scaler & Data ----------------------
try:
    scaler = joblib.load('scaler.pkl')
    data = pd.read_csv('diabetes.csv')
except Exception as e:
    st.error(f"Error loading scaler or data: {e}")
    st.stop()

# ---------------------- Sidebar ----------------------
st.sidebar.title("📊 Data Insights")
st.sidebar.subheader("🧠 Select Model")

model_choice = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Logistic Regression", "Random Forest", "XGBoost")
)

model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

try:
    model = joblib.load(model_files[model_choice])
    st.sidebar.success(f"{model_choice} loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Optional Sidebar visualizations
if st.sidebar.checkbox("Show Data Head"):
    st.sidebar.dataframe(data.head())

if st.sidebar.checkbox("Class Distribution"):
    st.sidebar.bar_chart(data['Outcome'].value_counts())

if st.sidebar.checkbox("Show Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.sidebar.pyplot(fig)

# ---------------------- Main Tabs ----------------------
st.title("🯪 GlucoPredict – Early Diabetes Alert")
st.markdown(f'<p class="subtitle">Use manual input or upload a CSV to get predictions.<br>Currently Using Model: <b>{model_choice}</b></p>', unsafe_allow_html=True)

tabs = st.tabs(["📝 Manual Input", "📂 CSV Upload", "📜 User History"])
tab1, tab2, tab3 = tabs

# 🔁 Continue with your Tab logic exactly as you already have for tab1, tab2, tab3

# 🟢 You don’t need to change any logic below this, just make sure:
# - `st.set_page_config` is only at top
# - `st.stop()` is used right after `login_page()` to prevent loading other content unless logged in


# ---------- Tab 1: Manual Input ----------
with tab1:
    st.subheader("🧾 Enter your personal & medical details:")

    with st.form("input_form"):
        name = st.text_input("Full Name").strip()
        email = st.text_input("Email Address").strip()
        address = st.text_area("Address").strip()
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
                    payload = {
                        "Pregnancies": pregnancies,
                        "Glucose": glucose,
                        "BloodPressure": bp,
                        "SkinThickness": skin,
                        "Insulin": insulin,
                        "BMI": bmi,
                        "DiabetesPedigreeFunction": dpf,
                        "Age": age
                    }

                    response = requests.post("http://127.0.0.1:10000/predict", json=payload)

                    if response.status_code == 200:
                        prediction = response.json()['prediction']
                        st.subheader("📢 Prediction Result:")
                        if prediction == "Diabetic":
                            st.error("⚠️ You may have **diabetes**. Please consult a doctor.")
                        else:
                            st.success("✅ You are likely not diabetic. Keep it up! 💪")

                        # Save to history CSV
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
                            "Prediction": prediction
                        }])

                        if not os.path.exists(history_file) or os.stat(history_file).st_size == 0:
                            new_row.to_csv(history_file, index=False, quoting=csv.QUOTE_ALL)
                        else:
                            new_row.to_csv(history_file, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)

                        st.success("✅ User data and prediction saved to history.")
                        st.dataframe(new_row)

                        # Create PDF report
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, "Diabetes Prediction Report", 0, 1, 'C')
                        pdf.set_font("Arial", size=12)
                        pdf.ln(5)
                        pdf.cell(0, 10, f"Name: {name}", ln=True)
                        pdf.cell(0, 10, f"Email: {email}", ln=True)
                        pdf.cell(0, 10, f"Address: {address}", ln=True)
                        pdf.cell(0, 10, f"Blood Group: {blood_group}", ln=True)
                        pdf.cell(0, 10, f"Pregnancies: {pregnancies}", ln=True)
                        pdf.cell(0, 10, f"Glucose: {glucose}", ln=True)
                        pdf.cell(0, 10, f"Blood Pressure: {bp}", ln=True)
                        pdf.cell(0, 10, f"Skin Thickness: {skin}", ln=True)
                        pdf.cell(0, 10, f"Insulin: {insulin}", ln=True)
                        pdf.cell(0, 10, f"BMI: {bmi}", ln=True)
                        pdf.cell(0, 10, f"Diabetes Pedigree Function: {dpf}", ln=True)
                        pdf.cell(0, 10, f"Age: {age}", ln=True)
                        pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
                        pdf.ln(10)
                        pdf.cell(0, 10, "Thanks for using GlucoPredict!", ln=True)

                        pdf_bytes = pdf.output(dest='S').encode('latin1')

                        # Email PDF report function
                        def send_email_with_pdf(receiver_email, pdf_bytes, user_name):
                            try:
                                sender_email = "glucopredict@gmail.com"  # CHANGE THIS
                                app_password = "iwxr fvro riji wcvy"       # CHANGE THIS (Gmail App Password)

                                msg = EmailMessage()
                                msg['Subject'] = '🧾 Your Diabetes Prediction Report'
                                msg['From'] = sender_email
                                msg['To'] = receiver_email
                                msg.set_content(f"Hi {user_name},\n\nPlease find attached your diabetes prediction report.\n\nStay healthy!\n- GlucoPredict")

                                msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename='Diabetes_Report.pdf')

                                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                                    smtp.login(sender_email, app_password)
                                    smtp.send_message(msg)

                                return True
                            except Exception as e:
                                st.error(f"❌ Failed to send email: {e}")
                                return False

                        if send_email_with_pdf(email, pdf_bytes, name):
                            st.success("📩 PDF report sent to your email successfully!")
                        else:
                            st.warning("⚠️ PDF report could not be sent. Please check your email address.")

                    else:
                        st.error(f"Backend error: {response.json().get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error during prediction or communication with backend: {e}")

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

            required_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
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

                # Generate PDF of history table
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "User Prediction History", 0, 1, 'C')
                pdf.set_font("Arial", size=10)
                pdf.ln(4)

                page_width = pdf.w - 2 * pdf.l_margin
                col_width = page_width / len(history_df.columns)

                # Header row
                for col_name in history_df.columns:
                    pdf.cell(col_width, 10, col_name, border=1)
                pdf.ln()

                # Data rows (limit to 50)
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

                pdf_bytes = pdf.output(dest='S').encode('latin1')

                st.download_button(
                    "⬇ Download History PDF",
                    pdf_bytes,
                    "user_prediction_history.pdf",
                    "application/pdf"
                )
        except Exception as e:
            st.error(f"Error reading history file: {e}")
    else:
        st.warning("No history file found yet.")
