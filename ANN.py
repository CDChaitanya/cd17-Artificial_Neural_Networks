# -*- coding: utf-8 -*-

#PART 1 => DATA PREPROCESSING 

#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[: , 3:13].values
y = dataset.iloc[: , 13].values

#ENCODING THE CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LabelEncoder_X1 = LabelEncoder()              # FOR GEOGRAPHY
X[: , 1] = LabelEncoder_X1.fit_transform(X[: , 1])

LabelEncoder_X2 = LabelEncoder()        # THIS MUST BE BEFORE ONEHOTENCODER
X[: , 2] = LabelEncoder_X2.fit_transform(X[: , 2])    #FOR GENDER

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer( [("Geography" , OneHotEncoder() , [1] )]
                       , remainder ='passthrough')
X = ct.fit_transform(X)
#AVOIDING THE DUMMY VARIABLE TRAP 
X = X[: , 1:]

#SPLITING THE DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

#FEATURE SCALING (IMPORTANT IN NN)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PART 2 => ANN

import keras 
from keras.models import Sequential
from keras.layers import Dense

#INITIALIZING THE ANN
classifier = Sequential()

#ADDING INPUT LAYER & 1 HIDDEN LAYER
classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation='relu' , input_dim=11))

#ADDING 2 HIDDEN LAYER
classifier.add(Dense(units = 6 , kernel_initializer = 'uniform' , activation='relu' ))

#ADDING OUTPUT LAYER
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation='sigmoid'))

#COMPILING THE ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

#FITTING THE ANN TO TRAINING SET
classifier.fit(X_train, y_train , batch_size=10 , epochs=100)

#PART 3 => MAKING THE PREDICTION AND EVALUATING THE MODEL
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#MAKING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test , y_pred)

