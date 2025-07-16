import zipfile
import os
import joblib
import streamlit as st

# Ekstrak zip jika belum ada file pkl
if not os.path.exists("bike_app/bike_model.pkl"):
    with zipfile.ZipFile("bike_app/bike_model.zip", 'r') as zip_ref:
        zip_ref.extractall("bike_app")

# Load model
model = joblib.load("bike_app/bike_model.pkl")

# Contoh tampilan Streamlit
st.title("Prediksi Sepeda")
input_val = st.slider("Input Fitur", 0, 100)
pred = model.predict([[input_val]])
st.write("Hasil Prediksi:", pred)
