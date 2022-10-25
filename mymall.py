import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
mc = pd.read_csv('/content/mall_customer.csv')

mc.head()

mc.tail()

mc.describe()

mc.info()

x_mc = mc.drop(['Genre','CustomerID'], axis=1)  
x_mc

y_mc = mc['Genre']
y_mc

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_mc, y_mc)
Xtrain.head()

Xtrain.shape #75%

Xtest.shape #25%

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()                       
model.fit(Xtrain, ytrain)                 
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model) #accuracy is low

from sklearn.metrics import classification_report

print(classification_report(ytest, y_model))

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

#Confusion Matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
confusion_matrix = metrics.confusion_matrix(ytest, y_model)

print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_mc))

cm_display.plot()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, y_model))