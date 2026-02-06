import os

import streamlit as st
from st_pages import add_page_title

add_page_title()
#st.title("Welcome to the microbiota Analysis & AI Prediction toolbox ✨")
st.write("""
This website is designed to **facilitate** the data analysis of microbiota datasets. Specifically, the website is designed to compare two groups of microbiota datasets (e.g., case vs control, positive vs negative etc) that are identified through the metadata.
""")
st.write("""
It provides basic exploratory analysis and statistical tests tools. The website also provides various pre-processing tools and Artificial Intelligence (AI) models, including SVM, Logistic Regression, KNN, and XGBoost, to help discover potential patterns and insights in your data.
""")

st.write('Here is a preview of the website once the data has been uploaded: ')

image_path = 'assets/glimpse.png'
if os.path.exists(image_path):
    st.image(image_path)
else:
    st.warning(f"Image not found at: {image_path}")
