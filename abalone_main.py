import streamlit as st
import pandas as pd

st.write("""
# My Abalonely Age Prediction Application
""")


st.sidebar.header('User Inputs')
st.sidebar.subheader('Please enter your date:')

v_Sex = st.sidebar.radio('Sex', ['Male', 'Female', 'Infant'])

v_Length = st.sidebar.slider('Length', min_value=0.0, max_value=1.0, value=0.5)
v_Diameter = st.sidebar.slider(
    'Diameter', min_value=0.0, max_value=1.0, value=0.4)
v_Height = st.sidebar.slider('Length', min_value=0.0, max_value=1.0, value=0.1)
v_Whole_weight = st.sidebar.slider(
    'Whole weight', min_value=0.0, max_value=3.0, value=0.8)
v_Shucked_weight = st.sidebar.slider(
    'Shucked weight', min_value=0.0, max_value=2.0, value=0.3)
v_Viscera_weight = st.sidebar.slider(
    'Viscera weight', min_value=0.0, max_value=1.0, value=0.2)
v_Shell_weight = st.sidebar.slider(
    'Shell weight', min_value=0.0, max_value=2.0, value=0.2)

if v_Sex == 'Male':
    v_Sex = 'M'
elif v_Sex == 'Female':
    v_Sex = 'F'
else:
    v_Sex = 'I'

data = {
    'Sex': v_Sex,
    'Length': v_Length,
    'Diameter': v_Diameter,
    'Height': v_Height,
    'Whole_weight': v_Whole_weight,
    'Shucked_weight': v_Shucked_weight,
    'Viscera_weight': v_Viscera_weight,
    'Shell_weight': v_Shell_weight
}

df = pd.DataFrame(data, index=[0])

st.subheader('User Inputs:')
st.write(df)

# user input data cleaning and engineering
data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample], axis=0)
# st.write(df)

cat_data = pd.get_dummies(df[['Sex']])
X_new = pd.concat([cat_data, df], axis=1)

X_new = X_new[:1]
X_new = X_new.drop(columns=['Sex'])

st.subheader('Pre-Processed Inputs:')
st.write(X_new)

import pickle as pk

# load the model
loaded_norm = pk.load(open('normalization.pkl', 'rb'))

# apply the model
X_new = loaded_norm.transform(X_new)

st.subheader('Normalized Inputs:')
st.write(X_new)

# load the model
loaded_knn = pk.load(open('best_knn.pkl', 'rb'))

# apply the model
prediction = loaded_knn.predict(X_new)

st.subheader('Prediction:')
st.write(prediction)