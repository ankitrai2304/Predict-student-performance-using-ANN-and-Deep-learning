# ----------------------------------------------------------------------------
# Step 0: Import Necessary Libraries
# ----------------------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# ----------------------------------------------------------------------------
# Step 1: Define the same Neural Network Architecture
# ----------------------------------------------------------------------------
# This class MUST match the architecture used during training.
class StudentClassifier(nn.Module):
    def __init__(self, input_size):
        super(StudentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output_layer(x))
        return x

# ----------------------------------------------------------------------------
# Step 2: Load the Model and Preprocessor
# ----------------------------------------------------------------------------
# The input feature size must be known. From training, we know it's 56 after preprocessing.
INPUT_FEATURES = 56 

# Instantiate the model
model = StudentClassifier(INPUT_FEATURES)

# Load the trained weights
try:
    model.load_state_dict(torch.load('/workspaces/Predict-student-performance-using-ANN-and-Deep-learning/student_model.pth'))
    model.eval() # Set the model to evaluation mode
except FileNotFoundError:
    st.error("Model file ('/workspaces/Predict-student-performance-using-ANN-and-Deep-learning/student_model.pth') not found. Please run the training script first.")
    st.stop()


# Load the preprocessor
try:
    preprocessor = joblib.load('/workspaces/Predict-student-performance-using-ANN-and-Deep-learning/preprocessor.joblib')
except FileNotFoundError:
    st.error("Preprocessor file ('/workspaces/Predict-student-performance-using-ANN-and-Deep-learning/preprocessor.joblib') not found. Please run the training script first.")
    st.stop()


# ----------------------------------------------------------------------------
# Step 3: Create the Streamlit User Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title('🎓 Student Performance Predictor')
st.write("""
This app predicts whether a student will pass or fail based on their demographic, social, and school-related features. 
Please provide the student's information below.
""")

# Create columns for layout
col1, col2, col3 = st.columns(3)

# ---- Column 1: Personal Information ----
with col1:
    st.subheader("Personal Information")
    age = st.slider('Age', 15, 22, 17)
    sex = st.selectbox('Sex', ['Female', 'Male'])
    address = st.selectbox('Address', ['Urban', 'Rural'], help="U: Urban, R: Rural")
    famsize = st.selectbox('Family Size', ['Less or equal to 3', 'Greater Than 3'], help="LE3: Less or equal to 3, GT3: Greater than 3")
    Pstatus = st.selectbox('Parent\'s Cohabitation Status', ['Together', 'Apart'], help="T: Together, A: Apart")
    guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
    romantic = st.selectbox('In a Romantic Relationship?', ['no', 'yes'])

# ---- Column 2: Family & Home Environment ----
with col2:
    st.subheader("Family & Home Environment")
    Medu = st.slider("Mother's Education", 0, 4, 2, help="0: none, 4: higher education")
    Fedu = st.slider("Father's Education", 0, 4, 2, help="0: none, 4: higher education")
    Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
    Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
    famrel = st.slider('Quality of Family Relationships', 1, 5, 4)
    internet = st.selectbox('Internet Access at Home?', ['yes', 'no'])
    famsup = st.selectbox('Family Educational Support?', ['yes', 'no'])

# ---- Column 3: School & Social Life ----
with col3:
    st.subheader("School & Social Life")
    school = st.selectbox('School', ['GP', 'MS'])
    studytime = st.slider('Weekly Study Time', 1, 4, 2, help="1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs")
    failures = st.slider('Number of Past Class Failures', 0, 4, 0)
    activities = st.selectbox('Extra-curricular Activities?', ['yes', 'no'])
    higher = st.selectbox('Wants to Take Higher Education?', ['yes', 'no'])
    absences = st.slider('Number of School Absences', 0, 93, 2)
    goout = st.slider('Going Out with Friends', 1, 5, 3, help="1: very low, 5: very high")


# ----------------------------------------------------------------------------
# Step 4: Prediction Logic
# ----------------------------------------------------------------------------
if st.button('🔮 Predict Student Performance', type="primary"):
    # Create a dictionary of the input data
    input_data = {
        'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': 'course', 'guardian': guardian, 'traveltime': 2, 'studytime': studytime,
        'failures': failures, 'schoolsup': 'no', 'famsup': famsup, 'paid': 'no',
        'activities': activities, 'nursery': 'yes', 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': famrel, 'freetime': 3, 'goout': goout,
        'Dalc': 1, 'Walc': 1, 'health': 3, 'absences': absences
    }
    
    # NOTE: Some features were not included in the UI for simplicity.
    # Default values are provided for them: 'reason', 'traveltime', 'schoolsup', 'paid', 'nursery',
    # 'freetime', 'Dalc', 'Walc', 'health'.
    
    # Create a DataFrame from the dictionary
    # The columns MUST be in the same order as during training.
    # We can get this order from the preprocessor object itself.
    original_features = preprocessor.feature_names_in_
    input_df = pd.DataFrame([input_data], columns=original_features)

    # Preprocess the input data using the loaded preprocessor
    input_processed = preprocessor.transform(input_df)

    # Convert to a PyTorch tensor
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)

    # Make the prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()
        
    # Display the result
    st.subheader('Prediction Result')
    if probability > 0.5:
        st.success(f"The model predicts the student will **PASS** with a probability of {probability*100:.2f}%.")
    else:
        st.error(f"The model predicts the student will **FAIL** with a probability of {(1-probability)*100:.2f}%.")

