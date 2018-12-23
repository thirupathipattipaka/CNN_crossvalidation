# importing libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
#random seed
np.random.seed(7)

#reading the data
dataset = np.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

#train and test split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=7)


# create the cnn model

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fit the model
model.fit(X_train,y_train,validation_split=(X_test,y_test),epochs=150,batch_size=10)
