# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:46:22 2018

@author: pradeep
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:\git\Machine-Learning\Data-Pre-Processing\Data.csv')

#Take all the columns except last 
X = dataset.iloc[:,:-1].values

#Take only the 4th column 
y = dataset.iloc[:,3].values

# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Convert Categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc_X = LabelEncoder()
X[:,0] = labelEnc_X.fit_transform(X[:,0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()

labelEnc_y = LabelEncoder()
y = labelEnc_y.fit_transform(y)

# Splitting data  to Train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state =0)

# Scallng up the data

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =  sc_x.transform(x_test)
