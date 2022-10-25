import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

file = "mall_customer.csv"

mc = pd.read_csv(file)

st.header("KMeans Clustering for Mall Customer Dataset")
mc

features = ['Annual_Income_(k$)', 'Spending_Score']
mcF = mc[features]

st.write("Scatter plot for Annual Income and Spending Score")
fig = plt.figure(figsize=(8, 4))
plt.scatter(mcF['Annual_Income_(k$)'], mcF['Spending_Score'], c = "salmon");
st.pyplot(fig)

kmeans = KMeans(n_clusters=4)
kmeans.fit(mcF)
y_kmeans = kmeans.predict(mcF)

st.write("Scatter plot for Annual Income and Spending Score with centers")
fig = plt.figure(figsize=(8, 4))         
plt.scatter(mcF['Annual_Income_(k$)'], mcF['Spending_Score'], c=y_kmeans, s=50, cmap='inferno')
centers = kmeans.cluster_centers_
centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
st.pyplot(fig)