import time
import streamlit as st
from service import lr_model
from service import xgb_model


st.title('MBTI Post Classfier')

model_type = st.radio("Select a Model: ", ("XGBoost", "Logistic Regression"))

input_post = st.text_input(label=" ", placeholder='Enter your post')

prediction = None
if input_post != '':
    if model_type == "XGBoost":
        prediction = lr_model.predict(input_post)
    elif model_type == "Logistic Regression":
        prediction = xgb_model.predict(input_post)

if prediction is not None:
    st.text(f"👉 Your MBTI: {prediction}")
