# install libraries
# pip install streamlit
# run file -> streamlit run intro.py
import streamlit as st

# page title
st.title("Model Deployment with Streamlit")
st.header("Welcome to the Model Deployment Tutorial")

# add some text
st.write("""In this tutorial, we will learn how to deploy a machine learning model using Streamlit. 
Streamlit is an open-source app framework that allows you to create and share custom web apps for machine learning and data science projects with ease.""")

# take input from user
name = st.text_input("Enter your name:")
if st.button("Submit"):
    st.success(f"Hello, {name}! Welcome to the world of model deployment with Streamlit.")
    
import numpy as np
import pandas as pd   

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

age = st.slider("Select your age:", 0, 100, 25)
st.write(f"You selected: {age} years old") 


# slider and selecter 
value = st.slider("Select a value:", 0, 100, 50)
option = st.selectbox("Select city ", ["New York", "Los Angeles", "Chicago"])
st.write(f"You selected value: {value} and city: {option}")


if st.button("Click me"):
    st.balloons()