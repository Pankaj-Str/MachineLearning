import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import os
import matplotlib.pyplot as plt

# page configuration
st.set_page_config(page_title="Iris Classification", page_icon="🌸", layout="centered")

# title and description
st.title("Iris Flower Classification")
st.write("""In this app, we will classify iris flowers into three species: Setosa, Versicolor, and Virginica. 
We will use a Random Forest Classifier trained on the Iris dataset.""")

# side bar 
st.sidebar.header("Input Features")

# option load 
train_new = st.sidebar.checkbox("Train New Model",value=True)

@st.cache_data

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    target_names = iris.target_names
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # evaluation
    y_test_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    
    return model, target_names, acc , X_train.columns.tolist()



if train_new or not os.path.exists("iris_model.pkl"):
    model, target_names, acc , feature_names = load_data()
    # save model
    with open("iris_model.pkl", "wb") as f:
        pickle.dump((model, target_names, feature_names), f)
    st.sidebar.success(f"Model trained with accuracy: {acc:.2f}")
else:
    with open("iris_model.pkl", "rb") as f:
        model, target_names, feature_names= pickle.load(f)
    st.sidebar.info("Model loaded from file.")    
    

# main input 

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.5)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.5)
    

# create a input data frame
input_data = pd.DataFrame(
    {
        feature_names[0]: [sepal_length],
        feature_names[1]: [sepal_width],
        feature_names[2]: [petal_length],
        feature_names[3]: [petal_width]
    }
)    

# prediction button 
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    predicted_species = target_names[prediction]
    st.success(f"The predicted iris species is: {predicted_species}")
    
    # show probabilities
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame(model.predict_proba(input_data), columns=target_names)
    st.dataframe(prob_df.T)
    
    
# model performance visualization
st.subheader("Model Performance")
if 'acc' in locals():
    st.write(f"Model Accuracy: {acc:.2f}")
else:
    st.write("Model accuracy not available.")
        