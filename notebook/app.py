# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import pickle
# PAGE CONFIG
st.set_page_config(
    page_title="Housing Price Classification",
    page_icon="🏠",
    layout="wide"
)
# LOAD MODEL
model = pickle.load(
    open(
        "models/random_forest_classifier.pkl",
        "rb"
    )
)
# TITLE
st.markdown(
    """
    <h1 style='text-align: center; color: #1E3A8A;'>
    🏠 Housing Price Classification App
    </h1>
    """,
    unsafe_allow_html=True
)
st.write("Predict whether a house belongs to Low, Medium, or High price category.")
# SIDEBAR
st.sidebar.header("Enter House Details")
longitude = st.sidebar.number_input(
    "Longitude",
    value=-122.23
)
latitude = st.sidebar.number_input(
    "Latitude",
    value=37.88
)
housing_median_age = st.sidebar.number_input(
    "Housing Median Age",
    value=20
)
total_rooms = st.sidebar.number_input(
    "Total Rooms",
    value=1000
)
total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms",
    value=200
)
population = st.sidebar.number_input(
    "Population",
    value=800
)
households = st.sidebar.number_input(
    "Households",
    value=300
)
median_income = st.sidebar.number_input(
    "Median Income",
    value=3.5
)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    [0, 1, 2, 3, 4]
)
# CREATE DATAFRAME
input_data = pd.DataFrame({

    "longitude": [longitude],

    "latitude": [latitude],

    "housing_median_age": [housing_median_age],

    "total_rooms": [total_rooms],

    "total_bedrooms": [total_bedrooms],

    "population": [population],

    "households": [households],

    "median_income": [median_income],

    "ocean_proximity": [ocean_proximity]

})

# DISPLAY INPUT DATA

st.subheader("Input Data")

st.dataframe(input_data)

# PREDICTION

if st.button("Predict House Category"):

    prediction = model.predict(input_data)

    result = prediction[0]

    if result == 0:

        st.error("Predicted Category: LOW PRICE HOUSE")

    elif result == 1:

        st.warning("Predicted Category: MEDIUM PRICE HOUSE")

    else:

        st.success("Predicted Category: HIGH PRICE HOUSE")

# FOOTER

st.markdown("---")

st.markdown(
    """
    <center>
    Developed using Streamlit and Random Forest Classifier
    </center>
    """,
    unsafe_allow_html=True
)

