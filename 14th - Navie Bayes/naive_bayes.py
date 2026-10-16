# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv(r"D:\NIT\1. NIT_Batches\2. EVENING BATCH\N_Batch -- 6.00PM-- Dec 26\3. Oct\14th - Navie Bayes\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()  
classifier.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)  

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train, y_train) 
bias 




