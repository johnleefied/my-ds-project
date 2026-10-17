import streamlit as st
import pandas as pd
import numpy as np

st.title("🚀 My First Data Science App")
st.write("Welcome to my end-to-end machine learning project!")

# Generate sample chart
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Feature A', 'Feature B', 'Feature C']
)
st.line_chart(data)

slider_val = st.slider("Pick a number for testing:", 1, 100, 25)
st.success(f"You selected: {slider_val}")
