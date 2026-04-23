import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Student Placement Predictor - SoCS BINUS",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD MODEL (DI-CACHE)
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan file pkl ada di folder artifacts atau di folder yang sama
    model_path = 'artifacts/best_placement_pipeline.pkl'
    if not os.path.exists(model_path):
        # Jika tidak ada di artifacts, coba cari di root folder
        model_path = 'best_placement_pipeline.pkl'
        
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.title("🎓 Menu Utama")
    selection = st.radio(
        "Navigasi Halaman:",
        ["🏠 Beranda", "📊 Visualisasi Data", "🔮 Kalkulator Prediksi"]
    )
    st.markdown("---")
    st.info("Aplikasi Evaluasi Kesiapan Kerja Mahasiswa B28.")
    st.caption("Developed by: Kang Nicholas Darren Nugroho")

# ==========================================
# 4. HALAMAN 1: BERANDA
# ==========================================
if selection == "🏠 Beranda":
    st.title("Selamat Datang di Portal Kesiapan Kerja")
    st.subheader("School of Computer Science - Universitas Bina Nusantara")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Deskripsi Proyek
        Aplikasi ini menggunakan algoritma **Random Forest** untuk memprediksi potensi penempatan kerja mahasiswa 
        berdasarkan data akademik, pengalaman magang, dan keahlian teknis.
        
        **Instruksi:**
        1. Pilih menu **Visualisasi** untuk melihat tren data.
        2. Pilih menu **Kalkulator Prediksi** untuk mensimulasikan data Anda.
        """)
    with col2:
        st.success("Aplikasi siap digunakan!")

# ==========================================
# 5. HALAMAN 2: VISUALISASI DATA
# ==========================================
elif selection == "📊 Visualisasi Data":
    st.title("📊 Eksplorasi Data Mahasiswa")
    try:
        df = pd.read_csv('A.csv')
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Distribusi CGPA**")
            fig, ax = plt.subplots()
            sns.histplot(df['cgpa'], kde=True, color="#3498db", ax=ax)
            st.pyplot(fig)
        with col_b:
            st.write("**Status Penempatan**")
            fig, ax = plt.subplots()
            df['placement_status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal memuat data A.csv: {e}")

# ==========================================
# 6. HALAMAN 3: PREDIKSI (KALKULATOR)
# ==========================================
elif selection == "🔮 Kalkulator Prediksi":
    st.title("🔮 Prediksi Kesiapan Kerja")
    
    if model is None:
        st.error("File model 'best_placement_pipeline.pkl' tidak ditemukan!")
    else:
        with st.form("prediction_form"):
            st.markdown("### 📝 Input Data Mahasiswa")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                cgpa = st.number_input("IPK (CGPA)", 0.0, 10.0, 7.5)
                tenth = st.number_input("Nilai SMP (%)", 0.0, 100.0, 80.0)
                twelfth = st.number_input("Nilai SMA (%)", 0.0, 100.0, 80.0)
                backlogs = st.number_input("Backlogs", 0, 10, 0)
                attendance = st.slider("Kehadiran (%)", 0.0, 100.0, 85.0)
                
            with c2:
                coding = st.slider("Coding Skill", 1, 5, 3)
                comm = st.slider("Communication", 1, 5, 3)
                aptitude = st.slider("Aptitude", 1, 5, 3)
                study_h = st.slider("Jam Belajar/Hari", 0.0, 15.0, 5.0)
                sleep_h = st.slider("Jam Tidur/Hari", 0.0, 12.0, 7.0)
                
            with c3:
                internships = st.number_input("Magang Selesai", 0, 5, 1)
                projects = st.number_input("Project Selesai", 0, 10, 2)
                hackathons = st.number_input("Hackathon Ikut", 0, 10, 0)
                certs = st.number_input("Sertifikasi", 0, 10, 1)
                gender = st.selectbox("Gender", ["Male", "Female"])

            st.markdown("---")
            c4, c5, c6 = st.columns(3)
            with c4:
                branch = st.selectbox("Jurusan", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            with c5:
                income = st.selectbox("Income Level", ["Low", "Medium", "High"])
                extra = st.selectbox("Ekstrakurikuler", ["High", "Low", "None"])
            with c6:
                part_time = st.selectbox("Part-Time Job", ["Yes", "No"])
                internet = st.selectbox("Akses Internet", ["Yes", "No"])
                city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

            submitted = st.form_submit_button("🚀 Analisis Sekarang")

        if submitted:
            try:
                # MAPPING: Mengubah teks menjadi angka untuk fitur numerik (stress_level)
                stress_map = {"Low": 0, "Medium": 1, "High": 2}
                
                # Membangun DataFrame Input
                input_df = pd.DataFrame([{
                    'cgpa': cgpa,
                    'tenth_percentage': tenth,
                    'twelfth_percentage': twelfth,
                    'backlogs': backlogs,
                    'study_hours_per_day': study_h,
                    'attendance_percentage': attendance,
                    'projects_completed': projects,
                    'internships_completed': internships,
                    'coding_skill_rating': coding,
                    'communication_skill_rating': comm,
                    'aptitude_skill_rating': aptitude,
                    'hackathons_participated': hackathons,
                    'certifications_count': certs,
                    'sleep_hours': sleep_h,
                    'stress_level': stress_map.get(stress, 1), # Dikirim sebagai angka (float)
                    'gender': gender,
                    'branch': branch,
                    'part_time_job': part_time,
                    'family_income_level': income,
                    'city_tier': city,
                    'internet_access': internet,
                    'extracurricular_involvement': extra
                }])

                # Feature Engineering: Total Skill (Harus ada jika model dilatih dengannya)
                input_df['total_skill_rating'] = coding + comm + aptitude

                # Prediksi
                prediction = model.predict(input_df)[0]
                
                st.markdown("---")
                if prediction == 1 or str(prediction).lower() == 'placed':
                    st.success("✅ **HASIL: PLACED (Ditempatkan)**")
                    st.balloons()
                else:
                    st.warning("⚠️ **HASIL: NOT PLACED (Belum Ditempatkan)**")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Student Research Project - BINUS SoCS")
