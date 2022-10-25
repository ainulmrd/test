from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.header('SVM for Iris Dataset')
iris = sns.load_dataset('iris') 
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))

model = SVC()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)

a = accuracy_score(ytest, y_model) 
st.write("Accuracy score:", a)

cr = classification_report(ytest, y_model)
st.write("Classification report:", cr)

confusion_matrix(ytest, y_model)

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(ytest, y_model)
c = confusion_matrix
st.write("Confusion matrix:",c)
fig = plt.figure(figsize=(10, 4))
sns.heatmap(c, annot=True)
st.pyplot(fig)