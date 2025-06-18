import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64

# ---------- Page Config ----------
st.set_page_config(page_title="GlucoPredict – Early Diabetes Alert", layout="centered")

# ---------- CSS for Background and Text Styling ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                        url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        @media only screen and (max-width: 768px) {{
            .stApp {{
                background-attachment: scroll;
                background-position: center top;
                background-size: cover;
            }}
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
        """,
        unsafe_allow_html=True
    )

# ---------- Set background ----------
set_bg("Bg.png")

# ---------- Load Model, Scaler and Dataset ----------
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    data = pd.read_csv('diabetes.csv')
except Exception as e:
    st.error(f"Error loading model, scaler or dataset: {e}")
    st.stop()

# ---------- Sidebar Visualizations ----------
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
st.title("🯪 GlucoPredict – Early Diabetes Alert")
st.markdown('<p class="subtitle">Use manual input or upload a CSV to get predictions.</p>', unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📝 Manual Input", "📂 CSV Upload"])

# ---------- Tab 1: Manual Input ----------
with tab1:
    st.subheader("🧾 Enter your medical details:")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose", min_value=1, max_value=200, value=120)
            bp = st.number_input("Blood Pressure", min_value=1, max_value=140, value=70)
            skin = st.number_input("Skin Thickness", min_value=1, max_value=100, value=20)
        with col2:
            insulin = st.number_input("Insulin", min_value=1, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0, format="%.2f")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        try:
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            st.subheader("📢 Prediction Result:")
            if prediction == 1:
                st.error("⚠️ You may have **diabetes**. Please consult a doctor.")
                st.info(
                    "## 🯪 Precautions & Tips\n"
                    "- 🥦 Eat a high-fiber, low-carb diet\n"
                    "- 🏃 Regular exercise (30 min daily)\n"
                    "- 🚫 Avoid sugar-sweetened beverages\n"
                    "- ✅ Maintain healthy body weight\n"
                    "- 💧 Stay hydrated\n"
                    "- 🧘‍♀️ Manage stress & sleep well"
                )
            else:
                st.success("✅ You are likely not diabetic. Keep it up! 💪")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---------- Tab 2: CSV Upload ----------
with tab2:
    st.subheader("📂 Upload CSV File")
    st.markdown("Your CSV must have the following columns (without `Outcome` column):")
    st.code("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:", df.head())

            required_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
            if not all(col in df.columns for col in required_cols):
                missing_cols = list(set(required_cols) - set(df.columns))
                st.error(f"Missing columns in CSV: {missing_cols}")
            else:
                scaled_input = scaler.transform(df[required_cols])
                predictions = model.predict(scaled_input)
                df['Prediction'] = ['Diabetic' if p == 1 else 'Not Diabetic' for p in predictions]

                st.success("✅ Bulk prediction complete.")
                st.dataframe(df)

                csv_result = df.to_csv(index=False).encode()
                st.download_button(
                    label="📅 Download Prediction Results",
                    data=csv_result,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
