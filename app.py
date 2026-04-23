import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Work Readiness Predictor - BINUS SoCS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLE CSS CUSTOM ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.title("🎓 Navigasi Utama")
    
    selection = st.radio(
        "Pilih Menu:",
        ["🏠 Beranda", "📊 Visualisasi Data", "🔮 Prediksi Kesiapan Kerja"]
    )
    
    st.markdown("---")
    st.info("""
    **Project Info:**
    Evaluasi Tingkat Kesiapan Kerja Mahasiswa School of Computer Science (SoCS) B28.
    """)
    st.caption("Developed by: Kang Nicholas Darren Nugroho")

# ==========================================
# HALAMAN 1: BERANDA (HOME)
# ==========================================
if selection == "🏠 Beranda":
    st.title("Selamat Datang di Portal Evaluasi Kesiapan Kerja")
    st.subheader("School of Computer Science - Universitas Bina Nusantara")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Deskripsi Proyek
        Aplikasi ini dirancang untuk memetakan dan memprediksi potensi penempatan kerja mahasiswa berdasarkan berbagai parameter akademik dan keahlian teknis. 
        
        **Tujuan Utama:**
        - Memberikan gambaran pola distribusi akademik dengan peluang kerja.
        - Mengevaluasi pengaruh *soft skills* dan *hard skills* terhadap kesiapan kerja.
        - Membantu mahasiswa memahami aspek yang perlu dikembangkan sebelum lulus.
        
        ### Metodologi Model
        Kami menggunakan pendekatan **Ensemble Learning** yang mencakup:
        1. **Logistic Regression:** Sebagai baseline performa klasifikasi.
        2. **Random Forest:** Untuk mengidentifikasi *Feature Importance* (aspek penentu utama).
        3. **Gradient Boosting:** Untuk akurasi optimal dalam estimasi regresi.
        """)
    
    with col2:
        st.info("💡 **Tips:** Gunakan menu sidebar di sebelah kiri untuk berpindah halaman.")
        st.warning("Pastikan file model `.pkl` sudah tersedia di dalam folder `artifacts` agar fitur prediksi berfungsi.")

# ==========================================
# HALAMAN 2: VISUALISASI DATA
# ==========================================
elif selection == "📊 Visualisasi Data":
    st.title("📊 Eksplorasi Data Mahasiswa")
    st.write("Analisis distribusi variabel dari dataset historis mahasiswa.")
    
    try:
        # Gunakan path file dataset Anda
        df = pd.read_csv('A.csv') 
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Distribusi CGPA Mahasiswa**")
            fig, ax = plt.subplots()
            sns.histplot(df['cgpa'], kde=True, color="#3498db", ax=ax)
            st.pyplot(fig)
        
        with col_b:
            st.write("**Perbandingan Gender**")
            fig, ax = plt.subplots()
            df['gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax)
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error("File 'A.csv' tidak ditemukan di folder proyek Anda. Pastikan namanya sudah benar (A.csv) dan berada di folder yang sama dengan app.py.")
    except KeyError as e:
        st.error(f"Nama kolom tidak cocok! Kolom {e} tidak ditemukan di dalam data. Pastikan huruf besar/kecilnya sesuai dengan file asli Anda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat grafik: {e}")

# ==========================================
# HALAMAN 3: PREDIKSI
# ==========================================
elif selection == "🔮 Prediksi Kesiapan Kerja":
    st.title("🔮 Kalkulator Prediksi Kelulusan")
    st.write("Masukkan seluruh parameter untuk melihat hasil analisis model.")

    # Form Input yang sudah mencakup 22 Fitur
    with st.form("prediction_form"):
        st.markdown("### 📚 Akademik & Kebiasaan Belajar")
        c1, c2, c3 = st.columns(3)
        with c1:
            cgpa = st.number_input("IPK (CGPA)", min_value=0.0, max_value=10.0, value=7.5)
            tenth_perc = st.number_input("Nilai SMP (%)", min_value=0.0, max_value=100.0, value=80.0)
            twelfth_perc = st.number_input("Nilai SMA (%)", min_value=0.0, max_value=100.0, value=80.0)
            backlogs = st.number_input("Jumlah Mengulang Kelas", min_value=0, max_value=10, value=0)
        with c2:
            attendance = st.slider("Kehadiran (%)", 0.0, 100.0, 85.0)
            study_hours = st.slider("Jam Belajar Harian", 0.0, 15.0, 4.0)
            sleep_hours = st.slider("Jam Tidur Harian", 0.0, 12.0, 7.0)
            stress = st.selectbox("Tingkat Stres", ["Low", "Medium", "High"])
        with c3:
            coding = st.slider("Coding Skill (1-5)", 1, 5, 3)
            comm = st.slider("Communication Skill (1-5)", 1, 5, 3)
            aptitude = st.slider("Aptitude Skill (1-5)", 1, 5, 3)

        st.markdown("### 🏆 Pengalaman & Profil Ekstra")
        c4, c5, c6 = st.columns(3)
        with c4:
            projects = st.number_input("Jumlah Project", 0, 20, 2)
            internships = st.number_input("Jumlah Magang (Internships)", 0, 10, 1)
            hackathons = st.number_input("Jumlah Hackathon", 0, 10, 0)
            certs = st.number_input("Jumlah Sertifikasi", 0, 20, 1)
        with c5:
            branch = st.selectbox("Jurusan", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            extracurricular = st.selectbox("Ekstrakurikuler?", ["High", "Low", "None"])
            part_time = st.selectbox("Kerja Part-Time?", ["Yes", "No"])
        with c6:
            income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
            city = st.selectbox("Kota Domisili (Tier)", ["Tier 1", "Tier 2", "Tier 3"])
            internet = st.selectbox("Akses Internet", ["Good", "Average", "Poor", "Yes", "No"]) 
            
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Analisis Kesiapan Kerja")

    if submitted:
        # PENTING: Path file model
        model_path = 'artifacts/best_placement_pipeline.pkl' 
        
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                
                # =================================================================
                # 🛠️ PROSES TRANSLASI (MENGUBAH TEKS MENJADI ANGKA)
                # Catatan: Jika di Notebook Anda angkanya berbeda (misal Low=1, dst), 
                # silakan ganti angka 0, 1, 2 di bawah ini agar sesuai dengan notebook.
                # =================================================================
                map_level = {"Low": 0, "Medium": 1, "High": 2}
                map_extra = {"None": 0, "Low": 1, "High": 2}
                map_yesno = {"No": 0, "Yes": 1}
                
                # Menerapkan translasi pada variabel yang bermasalah
                stress_num = map_level.get(stress, 0)
                income_num = map_level.get(income, 0)
                extra_num = map_extra.get(extracurricular, 0)
                
                # (Opsional) Jika Part-Time juga berupa angka di Notebook, gunakan ini:
                # part_time = map_yesno.get(part_time, 0)
                
                # Membangun DataFrame persis dengan 22 fitur yang dibutuhkan model
                input_df = pd.DataFrame([{
                    'cgpa': cgpa,
                    'tenth_percentage': tenth_perc,
                    'twelfth_percentage': twelfth_perc,
                    'backlogs': backlogs,
                    'coding_skill_rating': coding,
                    'communication_skill_rating': comm,
                    'aptitude_skill_rating': aptitude,
                    'projects_completed': projects,
                    'branch': branch,
                    'gender': gender,
                    'part_time_job': part_time, # Biarkan teks jika di notebook di-OneHotEncode
                    
                    # ⚠️ MASUKKAN VARIABEL YANG SUDAH JADI ANGKA:
                    'extracurricular_involvement': extra_num, 
                    'stress_level': stress_num,
                    'family_income_level': income_num,
                    
                    'sleep_hours': sleep_hours,
                    'city_tier': city,
                    'certifications_count': certs,
                    'attendance_percentage': attendance,
                    'study_hours_per_day': study_hours,
                    'hackathons_participated': hackathons,
                    'internships_completed': internships,
                    'internet_access': internet
                }])
                
                # Eksekusi Feature Engineering otomatis (jika digunakan)
                input_df['total_skill_rating'] = input_df['coding_skill_rating'] + input_df['communication_skill_rating'] + input_df['aptitude_skill_rating']
                
                # Logika Prediksi
                prediction = model.predict(input_df)
                
                st.subheader("Hasil Analisis:")
                if prediction[0] == 1 or str(prediction[0]).lower() == 'placed':
                    st.success("✅ **Status: Prediksi Ditempatkan (Placed)**")
                    st.balloons()
                    st.write("Profil Anda sangat kompetitif untuk pasar kerja saat ini.")
                else:
                    st.warning("⚠️ **Status: Perlu Peningkatan (Not Placed Yet)**")
                    st.write("Fokuslah untuk meningkatkan keahlian teknis dan portofolio Anda.")
            else:
                st.error(f"Gagal: File model '{model_path}' tidak ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")