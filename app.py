import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("🚀 My First Data Science App")
st.write("Welcome to my end-to-end machine learning project!")

# 1. Generate sample data
np.random.seed(42)
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Feature A', 'Feature B', 'Feature C']
)

# 2. Interactive line chart with markers
fig = px.line(
    data,
    markers=True,
    title="Feature Trends Over Steps"
)
st.plotly_chart(fig, use_container_width=True)

slider_val = st.slider("Pick a number for testing:", 1, 100, 25)
st.success(f"You selected: {slider_val}")
