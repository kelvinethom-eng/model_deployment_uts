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
    page_title="Placement & Salary Predictor - B28",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD MODEL (DUAL MODELS)
# ==========================================
@st.cache_resource
def load_models():
    # Folder tempat menyimpan model
    base_path = 'artifacts/' if os.path.exists('artifacts/') else ''
    
    clf_path = os.path.join(base_path, 'best_placement_pipeline.pkl')
    reg_path = os.path.join(base_path, 'best_salary_regressor.pkl')
    
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
    
    return clf_model, reg_model

clf_model, reg_model = load_models()

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
    st.info("Aplikasi Evaluasi Kesiapan Kerja & Estimasi Gaji Mahasiswa B28.")
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
        Aplikasi ini menggunakan dua model Machine Learning sekaligus:
        1. **Classification (Random Forest):** Memprediksi probabilitas status penempatan kerja (*Placed/Not Placed*).
        2. **Regression (Random Forest):** Memberikan estimasi gaji (*Salary LPA*) berdasarkan profil Anda.
        
        **Cara Penggunaan:**
        - Masukkan data akademik dan teknis Anda di menu **Kalkulator Prediksi**.
        - Sistem akan menganalisis 22 parameter untuk memberikan hasil prediksi ganda.
        """)
    with col2:
        if clf_model and reg_model:
            st.success("✅ Semua model (Klasifikasi & Regresi) berhasil dimuat.")
        else:
            st.warning("⚠️ Beberapa model (.pkl) belum ditemukan di folder artifacts.")

# ==========================================
# 5. HALAMAN 2: VISUALISASI DATA
# ==========================================
elif selection == "📊 Visualisasi Data":
    st.title("📊 Eksplorasi Data Mahasiswa")
    try:
        df = pd.read_csv('A.csv')
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Hubungan CGPA vs Salary**")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='cgpa', y='salary_lpa', hue='placement_status', ax=ax)
            st.pyplot(fig)
        with col_b:
            st.write("**Distribusi Status Penempatan**")
            fig, ax = plt.subplots()
            df['placement_status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff'])
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal memuat data visualisasi (A.csv): {e}")

# ==========================================
# 6. HALAMAN 3: PREDIKSI (KALKULATOR GANDA)
# ==========================================
elif selection == "🔮 Kalkulator Prediksi":
    st.title("🔮 Prediksi Kesiapan & Estimasi Gaji")
    
    if clf_model is None or reg_model is None:
        st.error("Model .pkl tidak lengkap! Pastikan 'best_placement_pipeline.pkl' dan 'best_salary_regressor.pkl' ada di GitHub/folder artifacts.")
    else:
        with st.form("dual_prediction_form"):
            st.markdown("### 📝 Input Data Mahasiswa")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**Akademik**")
                cgpa = st.number_input("IPK (CGPA)", 0.0, 10.0, 7.5)
                tenth = st.number_input("Nilai SMP (%)", 0.0, 100.0, 80.0)
                twelfth = st.number_input("Nilai SMA (%)", 0.0, 100.0, 80.0)
                backlogs = st.number_input("Backlogs", 0, 10, 0)
                attendance = st.slider("Kehadiran (%)", 0.0, 100.0, 85.0)
                
            with c2:
                st.markdown("**Technical Skills**")
                coding = st.slider("Coding Skill", 1, 5, 3)
                comm = st.slider("Communication", 1, 5, 3)
                aptitude = st.slider("Aptitude", 1, 5, 3)
                study_h = st.slider("Jam Belajar/Hari", 0.0, 15.0, 5.0)
                sleep_h = st.slider("Jam Tidur/Hari", 0.0, 12.0, 7.0)
                
            with c3:
                st.markdown("**Pengalaman & Profil**")
                internships = st.number_input("Magang Selesai", 0, 5, 1)
                projects = st.number_input("Project Selesai", 0, 10, 2)
                hackathons = st.number_input("Hackathon", 0, 10, 1)
                certs = st.number_input("Sertifikasi", 0, 10, 1)
                gender = st.selectbox("Gender", ["Male", "Female"])

            st.markdown("---")
            c4, c5, c6 = st.columns(3)
            with c4:
                branch = st.selectbox("Jurusan", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            with c5:
                income = st.selectbox("Income Level", ["Low", "Medium", "High"])
                extra = st.selectbox("Extracurricular", ["High", "Low", "None"])
            with c6:
                part_time = st.selectbox("Part-Time Job", ["Yes", "No"])
                internet = st.selectbox("Akses Internet", ["Yes", "No"])
                city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

            submitted = st.form_submit_button("🚀 Analisis Sekarang")

        if submitted:
            try:
                # 1. MAPPING DATA (Agar sesuai dengan tipe data numerik di model)
                stress_map = {"Low": 0, "Medium": 1, "High": 2}
                
                # 2. MEMBANGUN DATAFRAME INPUT (22 Fitur)
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
                    'stress_level': stress_map.get(stress, 1), # Dikirim sebagai angka (float/int)
                    'gender': gender,
                    'branch': branch,
                    'part_time_job': part_time,
                    'family_income_level': income, # Di datatrain.py ini nominal (string ok)
                    'city_tier': city,
                    'internet_access': internet,
                    'extracurricular_involvement': extra # Di datatrain.py ini nominal (string ok)
                }])

                # 3. FEATURE ENGINEERING: Total Skill
                input_df['total_skill_rating'] = coding + comm + aptitude

                # 4. PREDIKSI KLASIFIKASI (Status)
                prediction_status = clf_model.predict(input_df)[0]
                
                # 5. PREDIKSI REGRESI (Gaji)
                prediction_salary = reg_model.predict(input_df)[0]
                
                # TAMPILKAN HASIL
                st.markdown("### 📊 Hasil Analisis")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction_status == 1 or str(prediction_status).lower() == 'placed':
                        st.success("✅ **STATUS: PLACED (Ditempatkan)**")
                        st.balloons()
                    else:
                        st.warning("⚠️ **STATUS: NOT PLACED (Belum Ditempatkan)**")
                
                with res_col2:
                    # Logika bisnis: Jika tidak placed, tampilkan estimasi 0 atau tetap tampilkan potensi?
                    # Umumnya jika placed, kita tampilkan estimasi gajinya.
                    final_salary = prediction_salary if (prediction_status == 1 or str(prediction_status).lower() == 'placed') else 0.0
                    st.metric(label="💰 Estimasi Gaji (Annual LPA)", value=f"{final_salary:.2f} LPA")
                    if final_salary > 0:
                        st.write("Potensi penghasilan Anda cukup kompetitif di pasar saat ini.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan prediksi: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Student Research Project - Universitas Bina Nusantara | B28 SoCS")
