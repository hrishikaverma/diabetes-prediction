from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# ✅ Load the scaler and model once
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('rf_model.pkl')  # Use Random Forest model
    print("✅ Model and Scaler loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model or scaler: {e}")
    exit()

# ✅ MongoDB Atlas connection URI
mongo_uri = "mongodb+srv://GlucoPredict:Gluco123@cluster1.3hlg9y3.mongodb.net/diabetes?retryWrites=true&w=majority"

# ✅ Connect to MongoDB
try:
    client = MongoClient(mongo_uri)
    db = client["diabetes"]
    collection = db["predictions"]
    print("✅ MongoDB client connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit()

@app.route('/', methods=['GET'])
def home():
    return "🚀 Welcome to GlucoPredict API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("📩 Request received with data:", data)

        # Validate required keys
        required_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(k in data for k in required_keys):
            return jsonify({"error": "Missing required input fields"}), 400

        # Extract & preprocess features
        features = np.array([data[k] for k in required_keys]).reshape(1, -1)
        print(f"🔢 Features extracted: {features}")

        scaled_features = scaler.transform(features)
        print(f"⚙️ Scaled features: {scaled_features}")

        # Prediction
        prediction = model.predict(scaled_features)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        print(f"🎯 Prediction result: {result}")

        # Save to MongoDB with timestamp
        record = {k: data[k] for k in required_keys}
        record["Prediction"] = result
        record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        insert_result = collection.insert_one(record)
        print(f"💾 Data inserted into MongoDB with ID: {insert_result.inserted_id}")

        return jsonify({"prediction": result})

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Start the Flask server
if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))  # fallback to port 5000
    print(f"🚀 Starting Flask server on http://127.0.0.1:{PORT}")
    app.run(debug=True, use_reloader=False, port=PORT)

# 🚨 Optional CORS Setup (uncomment if needed for cross-origin requests)
# from flask_cors import CORS
# CORS(app)
