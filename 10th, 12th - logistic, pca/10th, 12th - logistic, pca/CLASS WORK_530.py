# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT DATASET 
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\MyFile\0. DATASCIENCE PROJECT\15. Logistic regression with future prediction\Social_Network_Ads.csv")

# SPLIT WITH IV & DV
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# 75-25 TRAIN TEST SPLIT 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

# SCALE THE DATA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# TRAIN THE MODEL X_TRAIN & Y_TRAIN 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# create prediction for x_test 
y_pred = classifier.predict(X_test)

# confusion matrix now 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# MODEL ACCURACY
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#TRAINING ACCURACY
bias = classifier.score(X_train,y_train)
print(bias)

# TESTING ACCURACY
variance = classifier.score(X_test, y_test)
print(variance)

# MODLE PREDICTION ON FUTURE DATA OR UNSEEN DATA

dataset1 = pd.read_csv(r"C:\Users\Admin\Desktop\MyFile\0. DATASCIENCE PROJECT\15. Logistic regression with future prediction\Future prediction1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2, 3]].values

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final1.csv')

import os 
os.getcwd()

