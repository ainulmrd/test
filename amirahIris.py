import streamlit as st
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn import metrics

iris=sns.load_dataset('iris')
iris.head()

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(Xtrain, ytrain)
st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(Xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(Xtest, ytest)))

#Confusion matrix SVM: 
ymodel = model.predict(xtest)

a = accuracy_score(ytest, ymodel) 
st.write("Accuracy score:", a)

report = classification_report(ytest, ymodel)
st.write(report)

confusion_matrix(ytest, ymodel)

# Evaluate by means of a confusion matrix
confusion_matrix = metrics.confusion_matrix(ytest, ymodel)
cm = confusion_matrix
st.write(cm)
fig = plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True)
st.pyplot(fig)
