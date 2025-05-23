# %%
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

# %%
class IrisModel:
  def __init__(self):
    # 가중치와 편향 초기화
    self.w1 = tf.Variable(tf.random.normal([4,50]), dtype=tf.float32)
    self.b1 = tf.Variable(tf.zeros([50]), dtype=tf.float32)
    self.w2 = tf.Variable(tf.random.normal([50,30]), dtype=tf.float32)  # 올바름
    self.b2 = tf.Variable(tf.zeros([30]),dtype=tf.float32)
    self.w3 = tf.Variable(tf.random.normal([30,3]), dtype=tf.float32)
    self.b3 = tf.Variable(tf.zeros([3]),dtype=tf.float32)

  def __call__(self, x):
    x = tf.nn.sigmoid(tf.matmul(x,self.w1) + self.b1)
    x = tf.nn.sigmoid(tf.matmul(x,self.w2) + self.b2)
    return tf.nn.softmax(tf.matmul(x,self.w3) + self.b3)

# %% [markdown]
# IrisModel이라는 커스텀 딥러닝 모델을 클래스로 정의
# 이 클래스는 3층 MLP (다층 퍼셉트론) 구조
# tf.Variable(...)을 사용한 이유: 학습 도중 업데이트 가능한 파라미터
# 입력 차원: 4 (Iris 데이터의 특징 수)
# 1층 은닉층: 50개의 뉴런W1: (4, 50), b1: (50,)
# 2층 은닉층: 30개의 뉴런W2: (50, 30), b2: (30,)
# 출력층: 3개의 뉴런 (클래스 개수)W3: (30, 3), b3: (3,)
# _call__() 메서드를 구현
# model(x)처럼 호출 가능 (함수처럼 사용 가능)
# 입력 x (예: (None, 4))를 W1과 곱하고 b1을 더해 은닉층 1의 출력 계산
# 시그모이드 활성화 함수 적용
# 은닉층 2로 다시 입력 후 같은 방식 반복
# 마지막에 softmax로 다중 분류 확률값 출력

# %%
# 손실 함수 정의 (CrossEntropy)
def loss_fn(model,inputs,labels):
  predictions = model(inputs)
  labels_one_hot = tf.one_hot(labels, depth=3) # one hot encoding
  loss = tf.reduce_mean(tf.losses.categorical_crossentropy(labels_one_hot,predictions))
  return loss

optimizer  = tf.optimizers.Adam(learning_rate=0.001)

def train_step(model, inputs, labels):
  with tf.GradientTape() as tape:
    loss = loss_fn(model, inputs, labels)  # 손실 계산

  gradients = tape.gradient(loss, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3])  # 그래디언트 계산
  optimizer.apply_gradients(zip(gradients, [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]))  # 가중치 업데이트
  return loss



# %% [markdown]
# 입력 데이터 → 모델 → 예측값
# ↘ 손실 계산 (one-hot + crossentropy)
# ↘ GradientTape로 미분
# ↘ 옵티마이저가 가중치 업데이트
# loss = tf.reduce_mean(tf.losses.categorical_crossentropy(labels_one_hot, predictions))
# • categorical cross-entropy 손실 계산
# • one-hot 라벨 vs softmax 예측 확률 비교
# • reduce_mean: 배치 전체 평균 손실값
# optimizer = tf.optimizers.Adam(learning_rate=0.001)
# • Adam 옵티마이저 생성 (기본은 0.001 학습률)
# • 모델의 가중치를 업데이트하는 역할 담당
# def train_step(model, inputs, labels):한 배치에 대해 모델을 학습시키는 함수
# with tf.GradientTape() as tape:
# 자동 미분을 위한 컨텍스트, 이 블록 안의 연산을 테이프에 기록하여 나중에 미분 가능

# %%
# 정확도 계산 함수
def compute_accuracy(model,inputs,labels):
  predictions = model(inputs)
  predicted_class = tf.argmax(predictions, axis=1)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class, labels),tf.float32))
  return accuracy
# 데이터 로드
iris = datasets.load_iris()
X, y = iris.data,iris.target

# Train/test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# 데이터를 텐서로 변환
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

# %% [markdown]
# 정확도 계산 함수 정의
# def compute_accuracy(model, inputs, labels):
# 모델과 정답 라벨을 받아 **분류 정확도(accuracy)**를 계산
# predicted_class = tf.argmax(predictions, axis=1)
# 가장 확률이 높은 클래스 인덱스를 선택
# 예: [0.1, 0.7, 0.2] → 1
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class, labels), tf.float32))
# 예측값과 실제 정답을 비교해서 맞은 것만 1.0, 틀리면 0.0
# 평균을 내어 전체 정확도 산출
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
# X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
# NumPy 배열 → TensorFlow 텐서로 변환,
# 입력값(X)은 부동소수점(float32), 레이블(y)은 정수형(int64)
# TensorFlow는 학습 및 예측 시 내부적으로 Tensor 객체를 사용하므로 이 변환은 필수

# %%
model = IrisModel()

# 학습
num_epochs = 20
batch_size = 16
num_batches = int(np.ceil(len(X_train)/ batch_size))

for epoch in range(num_epochs):
  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]

    loss = train_step(model, X_batch, y_batch) # 학습 단계

  if epoch % 5 == 0:
    train_accuracy = compute_accuracy(model,X_train,y_train)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {train_accuracy:.4f}')
# 평가
test_accuracy = compute_accuracy(model,X_test,y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# %%
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

class IrisModel:
  def __init__(self):
    self.W1=tf.Variable(tf.random.normal([4,50]), dtype=tf.float32)
    self.b1=tf.Variable(tf.zeros([50]), dtype=tf.float32)
    self.W2=tf.Variable(tf.random.normal([50,30]), dtype=tf.float32)
    self.b2=tf.Variable(tf.zeros([30]), dtype=tf.float32)
    self.W3=tf.Variable(tf.random.normal([30,3]), dtype=tf.float32)
    self.b3=tf.Variable(tf.zeros([3]), dtype=tf.float32)

  def __call__(self, x):
    x=tf.nn.sigmoid(tf.matmul(x, self.W1)+self.b1)
    x=tf.nn.sigmoid(tf.matmul(x, self.W2)+self.b2)
    return tf.nn.softmax(tf.matmul(x, self.W3)+self.b3)

def loss_fn(model, inputs, labels):
  predictions = model(inputs)
  labels_one_hot = tf.one_hot(labels, depth=3)
  loss = tf.reduce_mean(tf.losses.categorical_crossentropy(labels_one_hot, predictions))
  return loss

optimizer = tf.optimizers.Adam(learning_rate=0.001)

def train_step(model, inputs, labels):
  with tf.GradientTape() as tape:
    loss = loss_fn(model, inputs, labels)
  gradients = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3])
  optimizer.apply_gradients(zip(gradients, [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]))
  return loss


def compute_accuracy(model, inputs, labels):
  predictions = model(inputs)
  predicted_class=tf.argmax(predictions, axis=1)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_class, labels), tf.float32))
  return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

model = IrisModel()

num_epochs =20
batch_size=16
num_batches = int(np.ceil(len(X_train)/batch_size))

for epoch in range(num_epochs):
  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]

    loss = train_step(model, X_batch, y_batch)

  if epoch % 5 == 0:
    train_accuracy = compute_accuracy(model, X_train, y_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss:{loss:.4f}, Accuracy:{train_accuracy:.4f}")

    test_accuracy = compute_accuracy(model, X_test, y_test)
    print(f"Test Accuracy: { test_accuracy: .4f}")


