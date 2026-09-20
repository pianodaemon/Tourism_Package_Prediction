import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="j4nusx/tourism-model", filename="xgb_tourism_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package")

# Collect user inputs
Age                      = st.slider("Age", 18, 70, 30)
TypeofContact            = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier                 = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch          = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation               = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender                   = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting   = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups        = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched           = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar    = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus            = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips            = st.slider("Number of Trips", 1, 20, 3)
Passport                 = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore   = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar                   = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
Designation              = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome            = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

# ----------------------------
# Prepare input data (16 Selected Features)
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact_Self Enquiry': 1 if TypeofContact == "Self Enquiry" else 0,
    'Occupation_Large Business': 1 if Occupation == "Large Business" else 0,
    'Gender_Male': 1 if Gender == "Male" else 0,
    'MaritalStatus_Single': 1 if MaritalStatus == "Single" else 0,
    'MaritalStatus_Unmarried': 1 if MaritalStatus == "Unmarried" else 0,
    'Designation_Executive': 1 if Designation == "Executive" else 0,
    'Designation_Senior Manager': 1 if Designation in ["Sr. Manager", "Senior Manager"] else 0
}])

# Set the classification threshold
classification_threshold = 0.40

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    result = "will purchase the travel package" if pred == 1 else "is unlikely to purchase"
    st.write(f"Prediction: Customer {result}")
