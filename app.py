import streamlit as st
import pandas as pd
from prediction import make_prediction

# Kamus label ↔ kode numerik
gender = {0: "Perempuan", 1: "Laki-laki"}
scholarship = {0: "Tidak", 1: "Ya"}
debtor = {0: "Tidak", 1: "Ya"}
tuition_fees = {0: "Tidak", 1: "Ya"}
application_mode = {
    1: "Umum - Tahap 1", 2: "Regulasi 612/93", 5: "Khusus - Azores", 7: "Lulusan lain",
    10: "Regulasi 854-B/99", 15: "Internasional", 16: "Khusus - Madeira",
    17: "Umum - Tahap 2", 18: "Umum - Tahap 3", 26: "Rencana Lain", 27: "Lembaga Lain",
    39: "Di atas 23 tahun", 42: "Transfer", 43: "Ganti jurusan", 44: "Diploma teknis",
    51: "Ganti institusi", 53: "Diploma pendek", 57: "Ganti institusi (Internasional)"
}

# Balikan untuk konversi label → angka
inv_gender = {v: k for k, v in gender.items()}
inv_scholarship = {v: k for k, v in scholarship.items()}
inv_debtor = {v: k for k, v in debtor.items()}
inv_tuition = {v: k for k, v in tuition_fees.items()}
inv_app_mode = {v: k for k, v in application_mode.items()}

# Judul Aplikasi
st.title("🎓 Prediksi Dropout Mahasiswa")

# Pilihan input
input_mode = st.radio("Pilih metode input data:", ["Upload File CSV", "Input Manual"])

# ----------------------------- CSV Upload -----------------------------
if input_mode == "Upload File CSV":
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, delimiter=';')
            labels, probas = make_prediction(data)

            result_df = data.copy()
            result_df["Predicted_Status"] = labels
            result_df["Prob_Tidak_Dropout"] = probas[:, 0]
            result_df["Prob_Dropout"] = probas[:, 1]

            st.success("✅ Prediksi berhasil!")
            st.write(result_df)

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil", csv, "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi error saat memproses file: {e}")

# ----------------------------- Manual Input -----------------------------
else:
    st.subheader("📝 Input Manual Data Mahasiswa")

    # Tampilkan input dalam bentuk label teks
    application_label = st.selectbox("Application Mode", list(application_mode.values()))
    debtor_label = st.selectbox("Debtor", list(debtor.values()))
    tuition_label = st.selectbox("Tuition Fees Up To Date", list(tuition_fees.values()))
    gender_label = st.selectbox("Gender", list(gender.values()))
    scholarship_label = st.selectbox("Scholarship Holder", list(scholarship.values()))
    age = st.number_input("Age at Enrollment", min_value=15, max_value=100)
    cu1_approved = st.number_input("1st Sem Approved Units", min_value=0)
    cu1_grade = st.number_input("1st Sem Grade", min_value=0.0)
    cu2_approved = st.number_input("2nd Sem Approved Units", min_value=0)
    cu2_grade = st.number_input("2nd Sem Grade", min_value=0.0)

    if st.button("🔍 Prediksi"):
        try:
            # Konversi label kembali ke angka
            input_data = {
                'Application_mode': inv_app_mode[application_label],
                'Debtor': inv_debtor[debtor_label],
                'Tuition_fees_up_to_date': inv_tuition[tuition_label],
                'Gender': inv_gender[gender_label],
                'Scholarship_holder': inv_scholarship[scholarship_label],
                'Age_at_enrollment': age,
                'Curricular_units_1st_sem_approved': cu1_approved,
                'Curricular_units_1st_sem_grade': cu1_grade,
                'Curricular_units_2nd_sem_approved': cu2_approved,
                'Curricular_units_2nd_sem_grade': cu2_grade,
            }

            df_manual = pd.DataFrame([input_data])
            labels, probas = make_prediction(df_manual)

            st.success("✅ Hasil Prediksi:")
            st.write(f"**Status:** {labels[0]}")
            st.write(f"**Probabilitas Tidak Dropout:** {probas[0][0]:.2f}")
            st.write(f"**Probabilitas Dropout:** {probas[0][1]:.2f}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")