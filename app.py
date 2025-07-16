import streamlit as st
import joblib
import numpy as np
import os
import gdown

MODEL_URL = "https://drive.google.com/drive/folders/1FJW_BL2jsDR2XfAVXbTrSVjLlo0fSLPx?usp=sharing"
MODEL_PATH = "bike_model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = joblib.load("bike_model.pkl")

st.title("Prediksi Jumlah Penyewaan Sepeda")

hr = st.slider("Jam (0-23)", 0, 23, 12)
temp = st.slider("Suhu (0-1)", 0.0, 1.0, 0.5)
hum = st.slider("Kelembapan (0-1)", 0.0, 1.0, 0.5)

if st.button("Prediksi"):
    input_data = np.array([[hr, temp, hum]])
    prediction = model.predict(input_data)
    st.success(f"Prediksi jumlah sepeda: {int(prediction[0])}")
