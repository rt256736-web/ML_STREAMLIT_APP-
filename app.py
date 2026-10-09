import streamlit as st
import joblib
import numpy as np

# Load the trained ML model
model = joblib.load("model.pkl")

# Title
st.title("🌸 Iris Flower Prediction")

st.write(
    "Enter the flower measurements below "
    "to predict the Iris flower species."
)

# Input 1
sepal_length = st.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.0
)

# Input 2
sepal_width = st.slider(
    "Sepal Width (cm)",
    2.0,
    5.0,
    3.0
)

# Input 3
petal_length = st.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.5
)

# Input 4
petal_width = st.slider(
    "Petal Width (cm)",
    0.1,
    3.0,
    0.2
)

# Prediction button
if st.button("🔮 Predict"):

    # Prepare input
    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Flower names
    flower_names = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    result = flower_names[prediction[0]]

    # Display result
    st.success(f"🌸 Prediction: {result}")