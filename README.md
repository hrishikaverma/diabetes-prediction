# Diabetes Prediction Project

## Project Overview
Diabetes Prediction Project ek machine learning based application hai jo user ke medical parameters ke basis par predict karta hai ki usse diabetes ho sakta hai ya nahi. Isme PIMA Indians Diabetes Dataset use kiya gaya hai, jisme health related important features hain jaise glucose level, blood pressure, BMI, insulin, etc.

Ye project data cleaning, model training, aur prediction ke liye Streamlit web app interface provide karta hai jahan aap apne data input karke instant prediction le sakte hain. Saath hi, bulk data ke liye CSV upload kar ke bhi prediction kar sakte hain.

---

## Features
- **Data Preprocessing:** Missing ya zero values ko handle karta hai.
- **Machine Learning Models:** Logistic Regression, Random Forest, ya kisi aur model ka use karke accurate prediction.
- **User Interface:** Streamlit app jahan individual data input aur bulk CSV upload dono possible hain.
- **Precautions Section:** Diabetes na ho iske liye lifestyle tips aur health precautions diye gaye hain.
- **Visualization:** Data insights aur model performance ke liye charts (agar app me implemented ho).

---

## Dataset Description
PIMA Indians Diabetes Dataset from Kaggle use hua hai jisme ye features hain:

| Feature                | Description                          |
|------------------------|------------------------------------|
| Pregnancies            | Pregnancy count                    |
| Glucose                | Plasma glucose concentration      |
| BloodPressure          | Diastolic blood pressure           |
| SkinThickness          | Triceps skin fold thickness        |
| Insulin                | 2-Hour serum insulin                |
| BMI                    | Body mass index                    |
| DiabetesPedigreeFunction| Diabetes pedigree function          |
| Age                    | Age of the person                  |
| Outcome                | 1 = Diabetes, 0 = No Diabetes      |

---

## Tech Stack / Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - streamlit

Install karne ke liye:


pip install pandas numpy scikit-learn streamlit
Ya agar requirements.txt file ho to:


pip install -r requirements.txt
How to Run the Project
Repository clone karein:


git clone https://github.com/hrishikaverma/diabetes-prediction.git
Project folder me jaayein:


cd diabetes-prediction
Streamlit app run karein:


streamlit run app.py
Browser me app open hoga. Wahan aap manually apne health parameters daal kar diabetes prediction kar sakte hain. Bulk CSV upload ka option bhi available hai.

Precautions and Tips to Avoid Diabetes
Agar diabetes nahi hai to ye lifestyle follow karna chahiye:

Balanced diet lein, sugary aur processed food kam karein.

Regular exercise karein, daily kam se kam 30 minutes walk ya workout.

Weight control me rakhein, BMI normal range me ho.

Regular health check-ups karvate rahein.

Stress kam karein aur achi neend lein.

Alcohol aur smoking avoid karein.

Project Structure

diabetes-prediction/
│
├── app.py                # Main Streamlit app code
├── diabetes.csv          # Dataset file
├── model.py (optional)   # Model training and saving code (agar hai)
├── requirements.txt      # Required Python packages
└── README.md             # Ye file
Contribution
Aap is project me improvements ke liye pull requests bhej sakte hain, bugs report kar sakte hain, ya naye features add karne ke liye suggestions de sakte hain.

License
Is project ko MIT License ke under release kiya gaya hai. Aap freely use aur modify kar sakte hain.

Author
Hrishika Verma
GitHub Profile
LinkedIn Profile



