import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Placement Predictor B28", page_icon="🎓", layout="wide")

# --- LOAD MODEL (Cache agar cepat) ---
@st.cache_resource
def load_model():
    model_path = 'artifacts/best_placement_pipeline.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.title("🎓 Navigasi")
    menu = st.radio("Pilih Halaman:", ["🏠 Home", "📊 Visualisasi", "🔮 Prediksi"])
    st.markdown("---")
    st.caption("Developed by: Kang Nicholas Darren Nugroho")

# --- HALAMAN 1: HOME ---
if menu == "🏠 Home":
    st.title("Selamat Datang di Portal Kesiapan Kerja")
    st.write("Aplikasi ini memprediksi peluang penempatan kerja mahasiswa SoCS BINUS menggunakan Random Forest.")
    st.info("Gunakan menu sidebar untuk navigasi ke fitur prediksi.")

# --- HALAMAN 2: VISUALISASI ---
elif menu == "📊 Visualisasi":
    st.title("📊 Analisis Data Mahasiswa")
    try:
        df = pd.read_csv('A.csv')
        st.write("Distribusi CGPA Mahasiswa")
        fig, ax = plt.subplots()
        sns.histplot(df['cgpa'], kde=True, ax=ax)
        st.pyplot(fig)
    except:
        st.warning("File A.csv tidak ditemukan untuk visualisasi.")

# --- HALAMAN 3: PREDIKSI ---
elif menu == "🔮 Prediksi":
    st.title("🔮 Prediksi Kesiapan Kerja")
    
    if model is None:
        st.error("Model .pkl tidak ditemukan di folder artifacts!")
    else:
        with st.form("main_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📚 Akademik")
                cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
                tenth = st.number_input("Nilai SMP (%)", 0.0, 100.0, 80.0)
                twelfth = st.number_input("Nilai SMA (%)", 0.0, 100.0, 80.0)
                backlogs = st.number_input("Backlogs", 0, 10, 0)
                attendance = st.slider("Kehadiran (%)", 0.0, 100.0, 85.0)
                study_h = st.slider("Jam Belajar/Hari", 0.0, 15.0, 5.0)

            with col2:
                st.subheader("💻 Skill & Activity")
                coding = st.slider("Coding Skill", 1, 5, 3)
                comm = st.slider("Communication", 1, 5, 3)
                aptitude = st.slider("Aptitude", 1, 5, 3)
                internships = st.number_input("Magang", 0, 5, 1)
                projects = st.number_input("Projects", 0, 10, 2)
                hackathons = st.number_input("Hackathons", 0, 10, 0)
                certs = st.number_input("Sertifikasi", 0, 10, 1)

            with col3:
                st.subheader("🏫 Profil")
                branch = st.selectbox("Jurusan", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
                gender = st.selectbox("Gender", ["Male", "Female"])
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                income = st.selectbox("Income Level", ["Low", "Medium", "High"])
                extra = st.selectbox("Extracurricular", ["High", "Low", "None"])
                part_time = st.selectbox("Part Time?", ["Yes", "No"])
                internet = st.selectbox("Internet?", ["Yes", "No"])
                city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
                sleep = st.slider("Jam Tidur", 0.0, 12.0, 7.0)

            submitted = st.form_submit_button("Analisis Sekarang")

        if submitted:
            # Bangun DataFrame (Pastikan urutan dan nama kolom sama dengan X_train)
            input_df = pd.DataFrame([{
                'cgpa': cgpa, 'tenth_percentage': tenth, 'twelfth_percentage': twelfth,
                'backlogs': backlogs, 'study_hours_per_day': study_h,
                'attendance_percentage': attendance, 'projects_completed': projects,
                'internships_completed': internships, 'coding_skill_rating': coding,
                'communication_skill_rating': comm, 'aptitude_skill_rating': aptitude,
                'hackathons_participated': hackathons, 'certifications_count': certs,
                'sleep_hours': sleep, 'stress_level': stress, 'gender': gender,
                'branch': branch, 'part_time_job': part_time, 'family_income_level': income,
                'city_tier': city, 'internet_access': internet, 'extracurricular_involvement': extra
            }])

            # Tambahkan fitur engineering jika diperlukan
            input_df['total_skill_rating'] = coding + comm + aptitude

            res = model.predict(input_df)[0]
            if res == 1 or str(res).lower() == 'placed':
                st.success("✅ Prediksi: PLACED (Ditempatkan)")
                st.balloons()
            else:
                st.warning("⚠️ Prediksi: NOT PLACED (Belum Ditempatkan)")   