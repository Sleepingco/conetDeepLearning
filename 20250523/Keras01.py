# %% [markdown]
# Sequential 클래스 이용

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()

# %% [markdown]
# 함수형 API 이용

# %%
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

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(300, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

# %%
model4 = Sequential()
model4.add(Dense(3, activation='softmax', input_shape=(4,)))
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

# %%
model4.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

# %%
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data,iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# %%
model.fit(X_train,y_train,epochs=200)

# %%
model4.fit(X_train,y_train,epochs=200)

# %%
model.evaluate(X_test,y_test)

# %%
model4.evaluate(X_test,y_test)

# %%
from tensorflow.keras import Sequential, layers

# %%
model = Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(5,activation='relu'))
model.add(layers.Dense(6,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
model.summary()

# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

# %%
model.fit(X_train,y_train,epochs=20)

# %%
model.evaluate(X_test,y_test)

# %% [markdown]
# 모델의 저장 및 예측

# %%
# 모델 저장
model.save("my_model.keras")
print("Model has saved")

# %%
import numpy as np
from tensorflow.keras.models import load_model

# load saved model
loaded_model = load_model("my_model.keras")
print("model succesfully loaded")

# insert new data (iris has 4 feature)
X_new = np.array([4.6,3.6,1.0,0.2]).reshape(1,-1) # (1,4) shape change

# predict
predictions = loaded_model.predict(X_new)
# predict result
print("예측된 확률분호:",predictions)
print("예측된 클래스:",np.argmax(predictions))

# %% [markdown]
# reshape(1, -1)
# 딥러닝 모델 입력 요구사항
# TensorFlow/Keras의 신경망은 입력 데이터를 2D 배열(배치 크기, 특성 수) 형태로 받음.
# 즉, (batch_size, feature_dim) 형태가 필요함.
# reshape(1, -1)을 통해 (1, 4)로 변환하면 배치 크기가 1인 샘플로 인식됨.
# 배치 처리(batch processing)
# 모델은 한 번에 여러 개의 샘플(batch)을 받을 수 있음.
# 예를 들어, reshape(3, -1)이라면 3개의 샘플을 동시에 예측
# 

# %%
X_new = np.array([[4.6, 3.6, 1.0, 0.2],
                  [5.0, 3.4, 1.2, 0.3],
                  [4.8, 3.1, 1.3, 0.2]]).reshape(3, -1)
print(X_new.shape)
# 출력: (3, 4)

# %% [markdown]
# Vs. reshape(-1, 4)와의 차이점
# reshape(1, -1) : 배치 크기 1인 샘플 → (1, 4)
# reshape(-1, 4) : 자동으로 샘플 개수를 맞춤 → N개의 샘플을 가진 (N, 4) 형태
# -1은 자동으로 샘플 개수를 결정함.

# %%
from types import prepare_class
import numpy as np
from tensorflow.keras.models import load_model

loaded_model = load_model("my_model.keras")
print("model loaded")
X_new = np.array([
[5.1, 3.5, 1.4, 0.2],
[6.3, 2.8, 5.1, 1.5],
[6.5, 3.0, 5.5, 1.8]
])

print("input data shape",X_new) #(3,4)
predictions = loaded_model.predict(X_new)
print("예측된 확률분포:\n",predictions)
prepare_classes = np.argmax(predictions,axis=1)
print("predeicted class",prepare_classes)

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.layers import concatenate,Activation

input = Input(shape=(4,))
dense1 = Dense(30,activation='relu')(input)
dense2 = Dense(20,activation='relu')(input)
dense3 = Dense(10,activation='relu')(input)
x = concatenate([dense1,dense2,dense3])
output = Dense(3,activation='softmax')(x)
model = Model(inputs=input,outputs=output)
model.summary()

# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

# %%
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data,iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# %%
model.fit(X_train,y_train,epochs=200)

# %%
# 모델 저장
model.save("my_model.keras")
print("Model has saved")

# %%
import numpy as np


X_new = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.3, 3.3, 4.7, 1.6],
    [5.8, 2.7, 4.1, 1.0],
    [7.1, 3.0, 5.9, 2.1],
    [6.5, 3.0, 5.2, 2.0],
    [5.0, 3.4, 1.6, 0.4],
    [6.1, 2.9, 4.7, 1.4],
    [6.7, 3.1, 5.6, 2.4],
    [5.4, 3.9, 1.7, 0.4]
])

from types import prepare_class
import numpy as np
from tensorflow.keras.models import load_model

loaded_model = load_model("my_model.keras")
print("model loaded")


print("input data shape",X_new) #(3,4)
predictions = loaded_model.predict(X_new)
print("예측된 확률분포:\n",predictions)
prepare_classes = np.argmax(predictions,axis=1)
print("predeicted class",prepare_classes)


# %%



