# -*- coding: utf-8 -*-
"""
Using K-means with 2 clusters on the Pima indians diabetes
data set.

"""
import pandas as pd
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score


diabetes = pd.read_csv('../../data/diabetes_kaggle/diabetes.csv')

diabetes_X = diabetes.drop('Outcome', axis = 1)
diabetes_y = diabetes['Outcome'].copy()

X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.1, shuffle = True, random_state = 33)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_means = cluster.KMeans(n_clusters = 2, n_init=100)
k_means.fit(X_train_scaled)

print(precision_score(y_train, k_means.labels_))