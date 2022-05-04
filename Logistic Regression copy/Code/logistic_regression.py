import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import random_projection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import seaborn as sns

##We are going to predict whether a flower is an iris-setosa or not

df = pd.read_csv('C:/Users/ezequ/Desktop/Facultad/Machine learning/MachineLearning/Logistic Regression copy/Data/iris.csv', header = None)

#We exclude the column names
df = df.iloc[1: , :]

print(df.tail())

##We need to replace the last column for the values 0 (if it is not iris-setosa) or 1 (if it is)

df.iloc[:, -1:] = np.where(df.iloc[:, -1:] == "Iris-setosa" ,1,0)
print(df.tail())

#df.iloc[:, -1:] = df.iloc[:, -1:].astype(float)

X= df.iloc[:, 1:-1].values
y= df.iloc[:, -1:].values

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(X_test)
print(y_test)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model= Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'SGD', loss= 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train_scaled,y_train, epochs = 64, verbose = 1)
J_list = model.history.history['loss']
plt.plot(J_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

J_train = model.evaluate(X_train_scaled, y_train)
J_test = model.evaluate(X_test_scaled, y_test)
print("J_train:")
print(J_train)
print("J_test:")
print(J_test)

predictions =np.where(model.predict(X_test_scaled) > 0.5, 1,0) 
y_hat_test = model.predict(X_test_scaled)
print(classification_report(y_test,predictions))

i = 0
while(i<predictions.size):
    print("Prediction:",predictions[i],"; Real value:",y_test[i])
    i=i+1

predictions =np.where(model.predict(X_train_scaled) > 0.5, 1,0) 
y_hat_train = model.predict(X_train_scaled)
print(classification_report(y_train,predictions))

i = 0
while(i<predictions.size/4):
    print("Prediction:",predictions[i],"; Real value:",y_train[i])
    i=i+1
