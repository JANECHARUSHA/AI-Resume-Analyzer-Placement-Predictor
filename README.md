# AI-Resume-Analyzer-Placement-Predictor
AI Resume Analyzer and Placement Prediction System using Machine Learning and Streamlit.  The system analyzes resumes, detects skills, predicts placement probability,  and provides career recommendations and job description matching.
# AI Resume Analyzer & Placement Prediction System

## Overview
The AI Resume Analyzer & Placement Prediction System is a machine learning-based application that analyzes resumes and predicts the probability of placement. The system extracts skills from a resume, evaluates the candidate profile, and provides recommendations to improve career opportunities.

## Features
- Resume Upload and Analysis (PDF)
- Automatic Skill Detection
- Resume Score Calculation
- Placement Probability Prediction
- Skill Gap Analysis
- Career Role Recommendation
- Company Recommendation
- Job Description Matching

## Technologies Used
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- PyPDF2
- Matplotlib

## Machine Learning Model
This project uses the **Random Forest Classifier** to predict placement probability.  
The model is trained using features such as:

- CGPA
- Number of Projects
- Internship Experience
- Number of Technical Skills

## Dataset
The dataset contains student information including CGPA, projects completed, internship experience, and technical skills. The model predicts whether a student has a high chance of placement.

## How to Run the Project

### 1 Install Dependencies
pip install -r requirements.txt

### 2 Train the Model
python train_model.py

### 3 Run the Application
streamlit run app.py

## Project Structure
AI_Resume_Placement_System  
│  
├── app.py  
├── train_model.py  
├── dataset.csv  
├── requirements.txt  
└── model.pkl  

## Future Improvements
- Advanced NLP-based resume parsing
- Larger dataset for better prediction accuracy
- Integration with real job portals
- More advanced job matching system
- 
## demo

<img width="1833" height="870" alt="image" src="https://github.com/user-attachments/assets/0c3ac2d8-5e92-4d52-9cd0-fde5986e3aed" />

## Author
Developed as a Machine Learning project for analyzing resumes and predicting placement probability.
