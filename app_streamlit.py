import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64

st.set_page_config(page_title="GlucoPredict – Early Diabetes Alert", layout="centered")

# ---------- CSS for background and text styling ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
        }}

        h1 {{
            font-size: 3rem !important;
            font-weight: bold;
            color: #ffffff !important;
            text-shadow: 2px 2px 5px #000000;
        }}

        .subtitle {{
            font-size: 90px;
            font-weight: 500;
            color: #e0e0e0;
            text-shadow: 1px 1px 2px #000000;
            margin-bottom: 25px;
        }}

        .stTabs [role="tab"] {{
            font-size: 1.2rem !important;
            font-weight: 600;
        }}

        h2, h3, h4, h5, h6 {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px #000000;
        }}

        label, .stNumberInput label, .stTextInput label {{
            color: #ffffff !important;
            font-weight: 600;
        }}

        input {{
            background-color: rgba(255, 255, 255, 0.15);
            color: #ffffff !important;
            border-radius: 5px;
        }}

        .stDownloadButton button, .stButton button {{
            background-color: #2e7d32;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }}

        .css-1d391kg p {{
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("Bg.png")  # Background image

# ---------- Load Model & Scaler ----------
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('diabetes.csv')

# ---------- Sidebar Visuals ----------
st.sidebar.title("📊 Data Insights")
if st.sidebar.checkbox("Show Data Head"):
    st.sidebar.dataframe(data.head())

if st.sidebar.checkbox("Class Distribution"):
    st.sidebar.bar_chart(data['Outcome'].value_counts())

if st.sidebar.checkbox("Show Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.sidebar.pyplot(fig)

# ---------- Main Title ----------
st.title("🩺 GlucoPredict – Early Diabetes Alert")
st.markdown('<p class="subtitle">Use manual input or upload a CSV to get predictions.</p>', unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📝 Manual Input", "📂 CSV Upload"])

# ---------- Tab 1: Manual Input ----------
with tab1:
    st.subheader("🧾 Enter your medical details:")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20)
            glucose = st.number_input("Glucose", 1, 200)
            bp = st.number_input("Blood Pressure", 1, 140)
            skin = st.number_input("Skin Thickness", 1, 100)
        with col2:
            insulin = st.number_input("Insulin", 1, 900)
            bmi = st.number_input("BMI", 1.0, 70.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
            age = st.number_input("Age", 1, 120)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(input_data)
        result = model.predict(scaled)[0]

        st.subheader("📢 Prediction Result:")
        if result == 1:
            st.error("⚠️ You may have **diabetes**. Please consult a doctor.")
            st.info("## 🩺 Precautions & Tips\n"
                    "- 🥦 Eat a high-fiber, low-carb diet\n"
                    "- 🏃 Regular exercise (30 min daily)\n"
                    "- 🚫 Avoid sugar-sweetened beverages\n"
                    "- ✅ Maintain healthy body weight\n"
                    "- 💧 Stay hydrated\n"
                    "- 🧘‍♀️ Manage stress & sleep well")
        else:
            st.success("✅ You are likely not diabetic. Keep it up! 💪")

# ---------- Tab 2: CSV Upload ----------
with tab2:
    st.subheader("📂 Upload CSV File")
    st.markdown("Your CSV must have the following columns (no `Outcome`):")
    st.code("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age", language='text')

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("✅ Uploaded Data Preview:", df.head())
        try:
            scaled_input = scaler.transform(df)
            predictions = model.predict(scaled_input)
            df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]
            st.success("✅ Bulk prediction complete.")
            st.dataframe(df)

            # Download link
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results", csv, "prediction_results.csv", "text/csv")
        except Exception as e:
            st.error("❌ Error: " + str(e))
