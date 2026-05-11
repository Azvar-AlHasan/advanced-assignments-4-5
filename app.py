import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Keselamatan Titanic", page_icon="🚢", layout="centered")

# Sidebar untuk Informasi Tugas
st.sidebar.title("Informasi")
st.sidebar.info(
    "**Dikembangkan oleh:** Muhamad Azvar Al Hasan\n\n"
    "**NIM:** 12409031030068\n\n"
    "**Kelas:** Sistem Informasi 4C\n\n"
    "**Kelompok:** 5"
)
st.sidebar.text("Tugas Weekly Class GDGOC AI/ML")

# Memuat model dengan cache agar tidak di-load ulang setiap kali ada perubahan input
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

# Tampilan Utama
st.title("🚢 Aplikasi Prediksi Keselamatan Titanic")
st.write("Masukkan detail penumpang di bawah ini untuk memprediksi peluang keselamatannya.")

# Form Input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3], help="1 = Kelas 1, 2 = Kelas 2, 3 = Kelas 3")
        sex = st.selectbox("Jenis Kelamin", ["male", "female"])
        embarked = st.selectbox("Pelabuhan Keberangkatan", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")

    with col2:
        sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", min_value=0, max_value=10, value=0)
        fare = st.number_input("Harga Tiket (Fare)", min_value=0.0, value=15.0, format="%.2f")

    submitted = st.form_submit_button("Prediksi")

# Logika Prediksi
if submitted:
    # Membuat DataFrame yang formatnya sesuai dengan data training
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Embarked': [embarked],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    # Melakukan prediksi
    prediction = model.predict(input_data)

    st.divider()
    st.subheader("Hasil Prediksi:")
    
    # Menampilkan hasil
    if prediction[0] == 1:
        st.success("🎉 Penumpang diprediksi **SELAMAT**.")
    else:
        st.error("💀 Penumpang diprediksi **TIDAK SELAMAT**.")