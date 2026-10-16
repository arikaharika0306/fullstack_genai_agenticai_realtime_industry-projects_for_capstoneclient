import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\NIT\1. NIT_Batches\2. EVENING BATCH\N_Batch -- 6.00PM-- Jan 27\3. Oct\8th, 9th - logistic, pca\2.LOGISTIC REGRESSION CODE\logit classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values 


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.25,
                                                    random_state=0) 

from sklearn.preprocessing import StandardScaler  
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train) 

print(classifier.get_params()) 

y_pred = classifier.predict(X_test)  


bias = classifier.score(X_train, y_train)
print(bias)

var = classifier.score(X_test,y_test)
print(var) 

# Confusion Matrix 




















