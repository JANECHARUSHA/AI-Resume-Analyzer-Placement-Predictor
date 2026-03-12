import streamlit as st
import pickle
import numpy as np
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl","rb"))
data = pd.read_csv("dataset.csv")

st.set_page_config(page_title="AI Resume Analyzer",layout="wide")

st.title("AI Resume Analyzer & Placement Predictor")

# Resume Upload
uploaded_file = st.file_uploader("Upload Resume (PDF)",type="pdf")

if uploaded_file is not None:

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text=""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # Skills list
    skills_list = [
    "python","java","c++","sql","machine learning","deep learning",
    "data science","pandas","numpy","tensorflow","pytorch",
    "power bi","tableau","excel","html","css","javascript",
    "react","node","flask","django","nlp"
    ]

    detected_skills=[]

    for skill in skills_list:
        if skill in text.lower():
            detected_skills.append(skill)

    skills_count=len(detected_skills)

    st.subheader("Detected Skills")
    st.write(detected_skills)

    # Project count
    projects_count=text.lower().count("project")

    # Internship detection
    internship=1 if "internship" in text.lower() else 0

    st.write("Projects:",projects_count)
    st.write("Internship:",internship)

    # CGPA input
    cgpa=st.number_input("Enter CGPA",0.0,10.0)

    # Resume score
    score=0
    score+=cgpa*5
    score+=skills_count*5
    score+=projects_count*10
    score+=internship*20

    if score>100:
        score=100

    st.subheader(f"Resume Score: {score}/100")

    # Prepare prediction input
    user_data=np.array([[cgpa,projects_count,internship,skills_count]])

    prediction=model.predict(user_data)

    probability=model.predict_proba(user_data)

    placement_prob=probability[0][1]*100

    st.subheader(f"Placement Probability: {placement_prob:.2f}%")

    if prediction[0]==1:
        st.success("High Chance of Placement")
    else:
        st.error("Low Chance of Placement")

    # Skill gap
    required_skills=["python","sql","machine learning","pandas"]

    missing=[]

    for skill in required_skills:
        if skill not in detected_skills:
            missing.append(skill)

    st.subheader("Skill Gap")
    st.write(missing)

    # Career role recommendation
    st.subheader("Career Role Recommendation")

    if "machine learning" in detected_skills:
        st.write("ML Engineer")
    elif "python" in detected_skills and "sql" in detected_skills:
        st.write("Data Analyst")
    elif "html" in detected_skills:
        st.write("Web Developer")
    else:
        st.write("Software Developer")

    # Company recommendation
    st.subheader("Recommended Companies")

    if skills_count>=5 and cgpa>=8:
        st.write("Google, Amazon, Microsoft")
    elif skills_count>=3:
        st.write("TCS, Infosys, Cognizant")
    else:
        st.write("Startup Companies")

# Job Description Matching

st.subheader("Job Description Matching")

job_desc = st.text_area("Paste Job Description Here")

if job_desc:

    matched = []

    for skill in detected_skills:
        if skill in job_desc.lower():
            matched.append(skill)

    match_score = (len(matched) / len(skills_list)) * 100

    st.write("Matched Skills:", matched)

    st.write(f"Resume Match Score: {match_score:.2f}%")

# Dashboard graphs

st.subheader("Dataset Dashboard")

fig,ax=plt.subplots(1,3,figsize=(18,5))

ax[0].scatter(data["CGPA"],data["Placement"])
ax[0].set_title("CGPA vs Placement")

ax[1].scatter(data["Skills"],data["Placement"])
ax[1].set_title("Skills vs Placement")

ax[2].scatter(data["Projects"],data["Placement"])
ax[2].set_title("Projects vs Placement")

st.pyplot(fig)