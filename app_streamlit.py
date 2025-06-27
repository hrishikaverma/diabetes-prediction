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

# Page Config
st.set_page_config(page_title="GlucoPredict – Early Diabetes Alert", layout="centered")

# Background Styling
def set_bg(image_file):
    if os.path.exists(image_file):
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

# Utility: Validate Email/Mobile
def is_valid_email_or_mobile(user_input):
    email_regex = r'^\S+@\S+\.\S+$'
    mobile_regex = r'^\d{10}$'
    return re.match(email_regex, user_input) or re.match(mobile_regex, user_input)

# OTP Handling
def send_otp():
    otp = random.randint(100000, 999999)
    st.session_state['generated_otp'] = str(otp)
    st.session_state['otp_sent_time'] = time.time()
    st.info(f"📩 Your OTP is: {otp}")  # For demo

# Login/Register
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

# Logout
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main App
def main_app():
    st.title("🪴 GlucoPredict Dashboard")
    st.success(f"👋 Welcome back, {st.session_state['name']}!")
    if st.button("Logout 🔒"):
        logout()

# Routing
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    set_bg("bg.jpg")
    login_page()
    st.stop()
else:
    set_bg("Bg.png")
    main_app()
