🎓 Student Performance Predictor using Deep Learning
📝 Overview
This project builds and deploys a deep learning model to predict whether a student will pass or fail based on a range of demographic, social, and school-related factors. The model is an Artificial Neural Network (ANN) built with PyTorch.

The project includes two main components:

A Python script (train.py) that handles data loading, preprocessing, model training, and evaluation.

An interactive web application (user.py) built with Streamlit that loads the trained model and allows users to make real-time predictions by inputting a student's information.

The model is trained on the "Student Performance Data Set" from the UCI Machine Learning Repository.

✨ Features
End-to-End ML Pipeline: Implements a full machine learning workflow from data cleaning to deployment.

Deep Learning Model: Utilizes a PyTorch-based Artificial Neural Network with dropout for regularization.

Interactive UI: A user-friendly web interface created with Streamlit for easy prediction.

Data Preprocessing: Uses scikit-learn pipelines to handle both numerical (scaling) and categorical (one-hot encoding) features automatically.

Performance Visualization: Generates plots for training/validation loss and accuracy, along with a confusion matrix for the test set.

🛠️ Technologies Used
Backend & Modeling: Python, PyTorch, scikit-learn, Pandas, NumPy

Frontend Web App: Streamlit

Data Visualization: Matplotlib, Seaborn

Model & Preprocessor Saving: Joblib

📂 Project Structure
.
├── 📄 student-mat.csv           # The raw dataset used for training
├── 🐍 student_predictor.py      # Main script to train, evaluate, and save the model
├── 🌐 app.py                    # Streamlit script to run the interactive web application
├── requirements.txt          # A list of all necessary Python packages
├── 🧠 student_model.pth          # (Generated) The saved, trained PyTorch model weights
├── 🔄 preprocessor.joblib        # (Generated) The saved scikit-learn data preprocessor
└── 📖 README.md                 # This file

🚀 How to Run
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.8 or higher

pip package manager

2. Setup & Installation
a. Clone the repository:

git clone <your-repository-url>
cd <your-repository-directory>

b. Create and activate a virtual environment (Recommended):

# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

c. Install the required packages:

pip install -r requirements.txt

3. Running the Project
The project has two main steps: training the model and then running the prediction app.

Step 1: Train the Model

First, you need to run the training script. This will process the student-mat.csv data and create the student_model.pth and preprocessor.joblib files.

python student_predictor.py

This script will print the training progress and display the evaluation plots.

Step 2: Launch the Web Application

Once the model and preprocessor files are generated, you can start the Streamlit web app.

streamlit run app.py

This command will open a new tab in your web browser with the interactive prediction interface. You can now input student data and get real-time predictions!