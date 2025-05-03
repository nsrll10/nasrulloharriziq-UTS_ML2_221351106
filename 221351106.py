import streamlit as st # type: ignore
import tensorflow  as tf # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="employee-future-prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Employee Future Prediction")
st.write("Input Data Karyawan")

# Form input pengguna
joining_year = st.selectbox("Tahun Bergabung ", ['2012', '2013', '2014', '2015', '2016', '2017', '2018'] )
experience = st.number_input("Pengalaman", min_value=0.0, max_value=7.0, value=3.0 )
age = st.number_input("Umur ", min_value=22.0, max_value=41.0, value=25.0)
payment_tier = st.selectbox("Payment Tier", ['1', '2', '3'])

if st.button("Prediksi"):
    # Preprocessing input
    input_data = np.array([[joining_year, experience, age, payment_tier]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)
    st.success("Hasil Prediksi: **{}**".format("Resign" if prediction == 1 else "Tidak Resign"))
