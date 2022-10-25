import streamlit as st 
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

st.header("Decision Tree for Iris Dataset")
iris = sns.load_dataset('iris')
iris

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)

fig = plt.figure(figsize=(14, 8))
clf.fit(xtrain, ytrain) 
tree.plot_tree(clf.fit(xtrain, ytrain) )
st.pyplot(fig)
cs = clf.score(xtest, ytest)
st.write("Accuracy score: ", cs)