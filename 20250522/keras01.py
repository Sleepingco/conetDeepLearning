# -*- coding: utf-8 -*-
"""Keras01.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZQTmScWn8t3Mao2a2mDTi1071UfvzxyB

Sequential 클래스 이용
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()

"""함수형 API 이용"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.layers import concatenate,Activation

input = Input(shape=(4,))
dense1 = Dense(50,activation='relu')(input)
dense2 = Dense(80,activation='relu')(input)
dense3 = Dense(30,activation='relu')(input)
x = concatenate([dense1,dense2,dense3])
output = Dense(3,activation='softmax')(x)
model = Model(inputs=input,outputs=output)
model.summary()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

model4 = Sequential()
model4.add(Dense(3, activation='softmax', input_shape=(4,)))
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

model4.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data,iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

model.fit(X_train,y_train,epochs=200)

model4.fit(X_train,y_train,epochs=200)

model.evaluate(X_test,y_test)

model4.evaluate(X_test,y_test)

from tensorflow.keras import Sequential, layers

model = Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(5,activation='relu'))
model.add(layers.Dense(6,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20)

model.evaluate(X_test,y_test)