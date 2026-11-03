
import streamlit as st
import joblib

# Load trained model
model = joblib.load("iris_model.pkl")

# App title
st.title("Iris Flower Prediction 🌸")

st.write("Enter the flower measurements:")

# Input features
sl = st.number_input("Sepal Length (cm)", min_value=0.0)
sw = st.number_input("Sepal Width (cm)", min_value=0.0)
pl = st.number_input("Petal Length (cm)", min_value=0.0)
pw = st.number_input("Petal Width (cm)", min_value=0.0)

# Prediction
if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Iris Flower: {flowers[pred[0]]}")
