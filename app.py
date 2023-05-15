# Streamlit app for Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris)

# Purpose: To demonstrate the use of Streamlit for data science
# Author: Sid van Vliet (github.com/sidvanvliet)
# Date: 2023-05-15

import streamlit as st
import pandas as pd
from sklearn import neighbors as skn

# Streamlit header and description
st.set_page_config(page_title='Iris classification', page_icon=':seedling:')

st.title('Iris classification :seedling:')
st.write('An application to demonstrate Streamlit in combination with a machine learning model.')
st.write(
    ':male-technologist: Sid van Vliet ([GitHub](https://github.com/sidvanvliet))  \n:school: HU University of Applied Sciences, Utrecht')

# Collect user input
st.sidebar.header("Properties")

sepal_length = st.sidebar.number_input(
    'Sepal length (cm)', min_value=0.0, max_value=10.0, value=5.0)

sepal_width = st.sidebar.number_input(
    'Sepal width (cm)', min_value=0.0, max_value=10.0, value=5.0)

petal_length = st.sidebar.number_input(
    'Petal length (cm)', min_value=0.0, max_value=10.0, value=5.0)

petal_width = st.sidebar.number_input(
    'Petal width (cm)', min_value=0.0, max_value=10.0, value=5.0)

# Setup
iris_data = pd.read_csv('data/Iris.csv')
X = iris_data.drop(['Id', 'Species'], axis=1)
y = iris_data['Species']

# Train model
model = skn.KNeighborsClassifier(n_neighbors=5)
model.fit(X.values, y)

# Define input as an array
input = [[sepal_length, sepal_width, petal_length, petal_width]]

# Specicies-image map
species = {
    'Iris-setosa': {
        'thumbnail': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1280px-Irissetosa1.jpg',
        'description': 'Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae.',
    },

    'Iris-versicolor': {
        'thumbnail': 'https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg',
        'description': 'Iris versicolor is also commonly known as the blue flag, harlequin blueflag, larger blue flag, northern blue flag, and poison flag, plus other variations of these names, and in Britain and Ireland as purple iris. It is a species of Iris native to North America, in the Eastern United States and Eastern Canada.',
    },

    'Iris-virginica': {
        'thumbnail': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1024px-Iris_virginica_2.jpg',
        'description': 'Iris virginica, with the common name Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial species of flowering plant in the Iridaceae family, native to central and eastern North America.',
    }
}

# Classify using user input
if st.sidebar.button("Classify", use_container_width=True):
    prediction = model.predict(input)

    # Select image and define caption
    thumbnail = species[prediction[0]]['thumbnail']

    # Render prediction
    st.divider()

    st.title(prediction[0].replace('-', ' '))
    st.image(
        thumbnail, width=400)

    st.write(species[prediction[0]]['description'])
