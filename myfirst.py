import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.header("My First Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))

st.header("Boxplot for Iris")
iris = sns.load_dataset('iris')
fig = plt.figure(figsize=(10, 4))
sns.boxplot(data=iris)
st.pyplot(fig)