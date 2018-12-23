import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
#random seed
np.random.seed(7)

#reading the data
dataset = np.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

# create the cnn model

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fit the model
model.fit(X,Y,validation_split=0.33,epochs=150,batch_size=10)
