import time

import streamlit as st

from service import lr_model as model


st.title('MBTI Post Classfier')

input_post = st.text_input(label="", placeholder='Input your post')

st.text(f"Your MBTI type: {model.predict(input_post)}")

# p = st.progress(0)
# for i in range(100):
#     time.sleep(0.05)
#     p.progress(i)