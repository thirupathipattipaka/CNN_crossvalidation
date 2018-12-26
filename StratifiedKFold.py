import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#random seed

np.random.seed(20)


#reading the data
dataset = np.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

#function for neural network

def nn_network():
    #start neural network
    model = Sequential()

    #add fully connected layer
    model.add(Dense(units=16,activation = 'relu',input_dim=8))
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units=1,activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


nn = KerasClassifier(build_fn=nn_network,
                     epochs =10,
                     batch_size=100,
                     verbose=0)


#evaluate the neural network using three fold cross validation
cross_val_score(nn,X,Y,cv=3)
