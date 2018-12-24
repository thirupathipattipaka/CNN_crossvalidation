# importing libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

#random seed
np.random.seed(7)

#reading the data
dataset = np.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)
cvscores = []

for train,test in kfold.split(X,Y):
    # create the cnn model
    model = Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #Fit the model
    model.fit(X[train],Y[train],epochs=150,batch_size=10,verbose=0)

    #evaluate the mode
    scores = model.evaluate(X[test],Y[test],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1]*100)

print("%.2f%%" % np.mean(cvscores))

