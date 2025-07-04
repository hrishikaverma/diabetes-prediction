import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import base64

# ✅ Page Config
st.set_page_config("Admin Panel – GlucoPredict", layout="wide")

# ✅ Hardcoded Admin Credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "gluco@admin123"

# ✅ MongoDB Setup
MONGO_URI = "mongodb+srv://GlucoPredict:Gluco123@cluster1.3hlg9y3.mongodb.net/diabetes?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["diabetes"]
collection = db["predictions"]

# ✅ Admin Login Function
def admin_login():
    st.title("🔐 GlucoPredict Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.success("✅ Login successful.")
            st.session_state['admin_logged_in'] = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")

# ✅ Admin Session Check
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

if not st.session_state['admin_logged_in']:
    admin_login()
    st.stop()

# ✅ Admin Dashboard
st.title("📊 GlucoPredict Admin Dashboard")

# ✅ Fetch All Data
try:
    data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
except Exception as e:
    st.error(f"❌ MongoDB Error: {e}")
    st.stop()

if data.empty:
    st.info("No prediction data available yet.")
    st.stop()

# ✅ Show available columns
st.caption(f"📌 Available Columns: {', '.join(data.columns)}")

# ✅ Summary Metrics
st.markdown("### 🔢 Prediction Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", len(data))
col2.metric("Diabetic Cases", (data['Prediction'] == "Diabetic").sum())
col3.metric("Non-Diabetic Cases", (data['Prediction'] == "Not Diabetic").sum())

# ✅ Filter/Search
st.markdown("### 🔍 Filter / Search Records")
col1, col2 = st.columns(2)
search_email = col1.text_input("Search by Email")
search_pred = col2.selectbox("Prediction Type", ["All", "Diabetic", "Not Diabetic"])

filtered_data = data.copy()

# ✅ Email Filter (with safety)
if 'Email' in filtered_data.columns:
    filtered_data['Email'] = filtered_data['Email'].astype(str)
    if search_email:
        filtered_data = filtered_data[filtered_data['Email'].str.contains(search_email, case=False, na=False)]
else:
    if search_email:
        st.warning("⚠️ 'Email' column not found in the data.")

# ✅ Prediction Filter
if 'Prediction' in filtered_data.columns and search_pred != "All":
    filtered_data = filtered_data[filtered_data['Prediction'] == search_pred]
elif search_pred != "All" and 'Prediction' not in filtered_data.columns:
    st.warning("⚠️ 'Prediction' column not found in the data.")

# ✅ Show Data
st.markdown("### 📋 Filtered Records")
if not filtered_data.empty:
    st.dataframe(filtered_data, use_container_width=True)

    # ✅ Download Filtered Data
    csv = filtered_data.to_csv(index=False).encode()
    st.download_button("⬇ Download Filtered CSV", csv, "admin_filtered_data.csv", "text/csv")
else:
    st.info("No records match the filter criteria.")

# ✅ Delete Record
st.markdown("### 🗑️ Delete Record (By Email and Timestamp)")
del_col1, del_col2 = st.columns(2)
del_email = del_col1.text_input("Email of record to delete")
del_timestamp = del_col2.text_input("Exact Timestamp (e.g., 2025-07-04 10:45:00)")

if st.button("❌ Delete Record"):
    if del_email and del_timestamp:
        result = collection.delete_one({"Email": del_email, "Timestamp": del_timestamp})
        if result.deleted_count:
            st.success("✅ Record deleted successfully.")
            st.rerun()
        else:
            st.warning("⚠️ No matching record found.")
    else:
        st.warning("⚠️ Please enter both Email and Timestamp.")

# ✅ Logout
if st.button("🚪 Logout"):
    del st.session_state['admin_logged_in']
    st.rerun()
