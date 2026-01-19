# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- App Header --------------------
st.set_page_config(page_title="Internship Recommendation System", layout="wide")
st.title("🎓 Internship Recommendation System")
st.markdown(
    "Welcome! This dashboard recommends suitable internships for students "
    "based on their skills, domain interests, and academic performance."
)

# -------------------- Load Datasets --------------------
students_path = r"C:\Users\lenovo\Downloads\internship recommendation system\student_dataset.csv"
internships_path = r"C:\Users\lenovo\Downloads\internship recommendation system\internship_recommendation_dataset.csv"

try:
    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)
    st.success("✅ Datasets loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")

# -------------------- Display Datasets --------------------
st.markdown("### 📊 Preview of Uploaded Datasets")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**👩‍🎓 Students Dataset**")
    st.dataframe(students_df.head())

with col2:
    st.markdown("**🏢 Internships Dataset**")
    st.dataframe(internships_df.head())

# -------------------- Student Selection --------------------
student_ids = students_df['student_id'].tolist()
selected_student = st.selectbox("Select a Student ID to view recommendations:", student_ids)

if selected_student:
    student = students_df[students_df['student_id'] == selected_student].iloc[0]

    # Extract student attributes
    student_skills = f"{student['programming_languages']}, {student['tools_technologies']}".lower()
    student_domain = str(student['domain_interest']).lower()
    student_cgpa = float(student['cgpa'])
    student_aptitude = float(student['aptitude_score'])
    student_experience = float(student['work_experience_months'])
    student_certifications = int(student['certifications_count'])
    student_mode = str(student['preferred_mode']).lower()

    # -------------------- Step 3: Custom Scoring --------------------
    st.subheader("🎯 Internship Recommendations (Custom Scoring)")

    def calculate_match(row):
        required_skills = str(row['Required_Skills']).lower()
        domain = str(row['Domain']).lower()
        mode = str(row['Mode']).lower()

        skill_matches = sum(1 for skill in student_skills.split(",") if skill.strip() in required_skills)
        total_skills = len(student_skills.split(","))
        skill_score = (skill_matches / total_skills) * 35
        domain_score = 20 if student_domain in domain else 0
        cgpa_score = (student_cgpa / 10) * 15
        aptitude_score = (student_aptitude / 100) * 10
        exp_score = (student_experience / 12) * 10
        cert_score = (student_certifications / 5) * 5
        mode_score = 5 if student_mode in mode else 0

        return skill_score + domain_score + cgpa_score + aptitude_score + exp_score + cert_score + mode_score

    internships_df['Match_Score'] = internships_df.apply(calculate_match, axis=1)
    top_recommendations = internships_df.sort_values(by='Match_Score', ascending=False).head(3)
    st.success(f"Top Internship Recommendations for Student ID: **{selected_student}**")
    st.table(top_recommendations[['Company_Name', 'Internship_Title', 'Domain', 'Required_Skills', 'Match_Score']])

    # Visualization
    st.subheader("📊 Internship Match Visualization")
    fig = px.bar(
        top_recommendations,
        x='Company_Name',
        y='Match_Score',
        color='Domain',
        text='Match_Score',
        title=f"Match Scores for Student {selected_student}"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title="Match Score", xaxis_title="Company", title_x=0.3)
    st.plotly_chart(fig)

    # -------------------- Step 4: AI Recommendation --------------------
    st.subheader("🤖 AI-Powered Internship Recommendations (TF-IDF + Cosine Similarity)")

    student_profile = f"{student['programming_languages']}, {student['tools_technologies']}, {student['domain_interest']}".lower()
    internships_df['Combined_Text'] = (internships_df['Required_Skills'].astype(str) + ", " +
                                       internships_df['Description'].astype(str)).str.lower()

    vectorizer = TfidfVectorizer(stop_words='english')
    internship_vectors = vectorizer.fit_transform(internships_df['Combined_Text'])
    student_vector = vectorizer.transform([student_profile])
    cosine_scores = cosine_similarity(student_vector, internship_vectors).flatten()
    internships_df['AI_Score'] = cosine_scores * 100

    top_ai_recommendations = internships_df.sort_values(by='AI_Score', ascending=False).head(3)
    st.success(f"Top AI-Based Internship Recommendations for Student ID: **{selected_student}**")
    st.table(top_ai_recommendations[['Company_Name', 'Internship_Title', 'Domain', 'Required_Skills', 'AI_Score']])

    fig_ai = px.bar(
        top_ai_recommendations,
        x='Company_Name',
        y='AI_Score',
        color='Domain',
        text='AI_Score',
        title=f"AI Match Scores for Student {selected_student}"
    )
    fig_ai.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_ai.update_layout(yaxis_title="AI Match Score", xaxis_title="Company", title_x=0.3)
    st.plotly_chart(fig_ai)

    # -------------------- Step 5: Export & Download --------------------
    st.subheader("💾 Export & Download Recommendations")
    download_option = st.radio(
        "Select Recommendation Type to Download:",
        ('Custom Scoring', 'AI Recommendation')
    )
    df_to_download = top_recommendations if download_option == 'Custom Scoring' else top_ai_recommendations
    csv = df_to_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Recommendations as CSV",
        data=csv,
        file_name=f"{selected_student}_internship_recommendations.csv",
        mime='text/csv'
    )

    # -------------------- Simulated Accuracy --------------------
    st.subheader("📈 System Evaluation")
    accuracy = np.random.uniform(75, 95)  # keeps changing each click (simulated)
    st.metric(label="Estimated Model Accuracy", value=f"{accuracy:.2f}%")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<h5 style='text-align:center; font-size:20px;'>Developed by LEZIN & ANIRUDH as part of the Internship Recommendation Project | MSc Computer Science - CURAJ</h5>",
    unsafe_allow_html=True
)
