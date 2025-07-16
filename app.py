import streamlit as st
import joblib
import numpy as np

model = joblib.load("bike_model.pkl")

st.title("Prediksi Jumlah Penyewaan Sepeda")

hr = st.slider("Jam (0-23)", 0, 23, 12)
temp = st.slider("Suhu (0-1)", 0.0, 1.0, 0.5)
hum = st.slider("Kelembapan (0-1)", 0.0, 1.0, 0.5)

if st.button("Prediksi"):
    input_data = np.array([[hr, temp, hum]])
    prediction = model.predict(input_data)
    st.success(f"Prediksi jumlah sepeda: {int(prediction[0])}")
