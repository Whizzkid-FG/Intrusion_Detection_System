import numpy as np 
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)

traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)

X,Y = traindata.iloc[:,1:42], traindata.iloc[:,0]
C,T = testdata.iloc[:,0] ,testdata.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X) #scale / normalize X.

traindata = np.array(trainX) #convert to np array()
trainlabel = np.array(Y) #convert to np array()

scaler = Normalizer().fit(T) #scale/ normalize x_te
testT = scaler.transform(T) 

testdata = np.array(testT) #convert to np array()
testlabel = np.array(C) #convert to np array()

# Do LR, SVM, NB, DT, RF.

#LR
classifiers = ["Logistic Regression ", "Naive Bayes", "Decision Trees", "Ensemble Model : Random Forest"]
classifier_deets = []

lr_model = LogisticRegression()
lr_model.fit(traindata, trainlabel)
expected = testlabel
lr_predicted = lr_model.predict(testdata)
lr_proba = lr_model.predict_proba(testdata)

y_train1 = expected
y_pred = lr_predicted

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

classifier_deets.append([accuracy, recall, precision, f1]) #acc, rec, pre, f1. Print in that order.

#Naive Bayes

nb_model = GaussianNB()
nb_model.fit(traindata, trainlabel)
expected = testlabel
nb_predicted = nb_model.predict(testdata)
nb_proba = nb_model.predict_proba(testdata)

y_train1 = expected
y_pred = nb_predicted

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

classifier_deets.append([accuracy, recall, precision, f1])

# DT

dt_model = DecisionTreeClassifier()
dt_model.fit(traindata, trainlabel)
expected = testlabel
dt_predicted = dt_model.predict(testdata)
dt_proba = dt_model.predict_proba(testdata)

y_train1 = expected
y_pred = dt_predicted

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

classifier_deets.append([accuracy, recall, precision, f1])

# RF

rf_model = RandomForestClassifier(n_estimators = 150)
rf_model.fit(traindata, trainlabel)
expected = testlabel
rf_predicted = rf_model.predict(testdata)
rf_proba = rf_model.predict_proba(testdata)

y_train1 = expected
y_pred = rf_predicted

accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

classifier_deets.append([accuracy, recall, precision, f1])

for i in range(len(classifiers)):
    print(classifiers[i],":")
    print("Accuracy : ", classifier_deets[i][0])
    print("Recall : ", classifier_deets[i][1])
    print("Precision : ", classifier_deets[i][2])
    print("F1 Score : ", classifier_deets[i][3])
    print("**************************************\n")