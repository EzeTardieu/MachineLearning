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

#We remove the first row of the dataframe from what we are taking, we just need the values, no the columns' names

X= df.iloc[:, 1:-1].values
y= df.iloc[:, -1:].values

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')

#Split the data in two: train data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(X_test)
print(y_test)

#This model needs the values to be in a scale

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("SCALED:",X_train_scaled)

#Creation of the model

model= Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'SGD', loss= 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train_scaled,y_train, epochs = 64, verbose = 1)
J_list = model.history.history['loss']
plt.plot(J_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

#We evaluate both train and test data

J_train = model.evaluate(X_train_scaled, y_train)
J_test = model.evaluate(X_test_scaled, y_test)
print("J_train:")
print(J_train)
print("J_test:")
print(J_test)

#Lastly, we have the predictions that our model does for the test data

predictions =np.where(model.predict(X_test_scaled) > 0.5, 1,0) 
y_hat_test = model.predict(X_test_scaled)
print(classification_report(y_test,predictions))

i = 0
while(i<predictions.size):
    print("Prediction:",predictions[i],"; Real value:",y_test[i])
    i=i+1



#And we added predictions on the train data aswell out of curiosity

predictions =np.where(model.predict(X_train_scaled) > 0.5, 1,0) 
y_hat_train = model.predict(X_train_scaled)
print(classification_report(y_train,predictions))

i = 0
while(i<predictions.size/4):
    print("Prediction:",predictions[i],"; Real value:",y_train[i])
    i=i+1

#It is important to say that this model was implemented on a dataset with very few but maybe obvious data.


sns.regplot(x=np.asarray(df.iloc[:, -2:-1].values).astype('float32'), y=y, data=df.iloc[:, :-2], logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
plt.show()