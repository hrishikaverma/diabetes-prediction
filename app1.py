from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient

app = Flask(__name__)

# 1. Load the scaler and model once
scaler = joblib.load('scaler.pkl')
model = joblib.load('rf_model.pkl')  # Trained model

# ✅ MongoDB Atlas connection URI (with password encoded)
mongo_uri = "mongodb+srv://GlucoPredict:Gluco123@cluster1.3hlg9y3.mongodb.net/diabetes?retryWrites=true&w=majority"

# ✅ Connect to MongoDB
client = MongoClient(mongo_uri)
print("✅ MongoDB client connected successfully")

# ✅ Select database and collection
db = client["diabetes"]
collection = db["predictions"]

@app.route('/', methods=['GET'])
def home():
    return "Welcome to GlucoPredict API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        print("📩 Backend hit hua with data:", data)

        # Extract features from incoming JSON
        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]
        print(f"🔢 Features received: {features}")

        # Preprocess features
        features_np = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_np)
        print(f"⚙️ Scaled features: {scaled_features}")

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        print(f"🎯 Prediction result: {result}")

        # Save to MongoDB
        insert_result = collection.insert_one({
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age'],
            "Prediction": result
        })
        print(f"💾 Data inserted with id: {insert_result.inserted_id}")

        return jsonify({"prediction": result})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False)
