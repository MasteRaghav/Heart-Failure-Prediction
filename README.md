# 🫀 Heart Failure Prediction System with 83% Accuracy

An interactive desktop application built with Tkinter and powered by Machine Learning to predict heart failure outcomes based on clinical data. The system provides an end-to-end workflow—from dataset loading and visualization to model training, evaluation, and live prediction—all in a user-friendly GUI.

# 🚀 Features
- 📂 Load CSV/Excel datasets
- 🧾 Data preview in GUI
- 📊 Model training using Random Forest
- 📈 Accuracy, Confusion Matrix, ROC Curve & Classification Report
- 📌 Feature Importance visualization
- 🔮 Single prediction input interface
- 💾 Save and load trained models
- 🎨 Modern, scrollable, and resizable interface

# 🧠 Machine Learning
- Algorithm: Random Forest Classifier
- Preprocessing: Standard Scaler
- Evaluation Metrics:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC-AUC Curve

# 📊 Dataset
File: heart_failure_clinical_records.csv
This dataset contains medical records of 299 patients with heart failure.

Target Variable:
- `DEATH_EVENT`: Indicates whether the patient died during the follow-up period (1 = yes, 0 = no)

Features:

# Feature                                  Description
- `Age`: age of the patient [years]
- `Sex`: sex of the patient [1 = Male, 0 = Female]
- `Anaemia`: Decrease of red blood cells or hemoglobin (boolean)
- `Creatinine_Phosphokinase`: CPK level in the blood
- `Cholesterol`: serum cholesterol [mm/dl]
- `Diabetes`: Whether the patient has diabetes
- `Ejection_Fraction`: Percentage of blood leaving the heart each beat
- `High_Blood_Pressure`: Whether the patient has hypertension
- `Platelets`: 	Platelet count in the blood
- `Serum_creatinine`: Level of serum creatinine in blood
- `Serum_sodium`: 	Level of serum sodium in blood
- `Smoking`: Whether the patient smokes
- `Time`: Follow-up period (in days)
- `Death_Event`:

Source:
This dataset is available publicly and has been used for heart failure prediction tasks in various research.


# 🛠️ Installation
1. Clone the repository:
```sh
git clone https://github.com/yourusername/heart-failure-prediction.git
cd heart-failure-prediction
```
2. Install dependencies:

```sh
pip install -r requirements.txt
```
3. Run the app:
```sh
python Heart_Failure_Prediction.py
```

# 📁 Model Saving & Loading
Trained models can be saved as .joblib files including:
- The Random Forest model
- The feature scaler
- Column structure
- These can be reloaded later for instant prediction without retraining.

# 📷 Screenshots
![Screenshot 2025-04-16 162755](https://github.com/user-attachments/assets/59b5d1dd-fa35-4d4e-b34d-ae969d9d7809)
![Screenshot 2025-04-16 163112](https://github.com/user-attachments/assets/25646d1e-4b68-432b-bdfa-82d6e49fb5d1)
![Screenshot 2025-04-16 163224](https://github.com/user-attachments/assets/2bc309cb-c627-4496-8c5a-9c7fb42a62a1)
