# -*- coding: utf-8 -*-
"""
Logistic regression on the Pima indians diabetes
data set.

"""
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score


diabetes = pd.read_csv('../../data/diabetes_kaggle/diabetes.csv')

diabetes_X = diabetes.drop('Outcome', axis = 1)
diabetes_y = diabetes['Outcome'].copy()

X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.1, shuffle = True, random_state = 33)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegressionCV(cv = 3)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_train_scaled)

print("Precision score: " + str(precision_score(y_train, y_pred)))
print("Recall score: " + str(recall_score(y_train, y_pred)))