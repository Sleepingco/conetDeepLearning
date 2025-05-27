# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


# %%
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0


# %% [markdown]
# Fashion MNIST 데이터를 로드하고 전처리.
# 
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# 
# 입력 데이터(x_train, x_test)의 형태를 (샘플 수, 28, 28, 1)로 reshape.  
# -1로 설정하면, 해당 차원의 크기는 자동으로 계산,  
# x_train 배열의 원래 크기를 유지하면서 3차원의 배열을 **4차원 배열로 생성**  
# (이미지 개수, 28, 28, 1)
# 
# x_train = x_train.astype(np.float32) / 255.0  
# x_test = x_test.astype(np.float32) / 255.0
# 
# np.float32로 형변환 한 뒤 255로 나누어 픽셀 값의 범위를 **0과 1 사이로 정규화**,  
# 모델의 입력 데이터를 더 잘 학습할 수 있도록.
# 

# %%
import matplotlib.pyplot as plt

# 25개의 이미지 출력
plt.figure(figsize=(6, 6))

for index in range(25):  # 25개 이미지 출력
    plt.subplot(5, 5, index + 1)  # 5행 5열
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
    # plt.title(str(t_train[index]))

plt.show()


# %% [markdown]
# 이미지 확인용으로 총 25개의 이미지를 출력
# 
# plt.figure(figsize=(6, 6)) 출력할 그림의 크기를 설정  
# figsize는 가로와 세로 크기를 인치 단위로 지정
# 
# for index in range(25):  
# plt.subplot(5, 5, index + 1) 출력할 그림의 위치를 지정.  
# 5행 5열의 그리드에서 현재 이미지의 위치를 지정.  
# index는 0부터 시작하므로 1을 더해준다.
# 
# plt.imshow(x_train[index], cmap='gray')  
# 현재 인덱스의 이미지를 그레이스케일로 출력.  
# x_train[index]는 해당 인덱스의 이미지 데이터를 나타낸다.
# 
# plt.axis('off')  
# 축을 표시하지 않도록 설정하는 부분
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Fashion MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 첫 번째 이미지 출력
image = x_train[0]
plt.imshow(image, cmap='gray')
plt.show()

# 행렬별 값 출력
for row in image:
    for pixel in row:
        print(f'{pixel:3d}', end=' ')
    print()


# %%
# 네 번째 이미지 출력
image = x_train[3]
plt.imshow(image, cmap='gray')
plt.show()

# 행렬별 값 출력
for row in image:
    for pixel in row:
        print(f'{pixel:3d}', end=' ')
    print()


# %%
cnn = Sequential()

cnn.add(Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, activation='relu'))
cnn.add(Conv2D(kernel_size=(3,3), filters=64, activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))


# %% [markdown]
# 입력으로부터 특성 추출을 위한 Convolutional 레이어와 pooling 레이어를 거친 후,  
# 특성 맵을 1차원으로 펼친 뒤  
# Fully Connected 레이어를 통해 분류 작업을 수행.  
# Softmax 레이어를 통해 클래스별 확률 값을 출력.
# 

# %% [markdown]
# cnn.add(Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, activation='relu'))  
# 첫 번째 Convolutional 레이어를 추가.  
# 입력 형태는 (28, 28, 1)이고, 커널 크기는 (3, 3)이며, 필터 개수는 32개.  
# 활성화 함수로는 ReLU를 사용.
# 
# cnn.add(Conv2D(kernel_size=(3,3), filters=64, activation='relu'))  
# 두 번째 Convolutional 레이어를 추가.  
# 커널 크기는 (3, 3)이며, 필터 개수는 64개.  
# 활성화 함수로는 ReLU를 사용.
# 
# cnn.add(MaxPool2D(pool_size=(2,2)))  
# Max Pooling 레이어를 추가. 풀 크기는 (2, 2).
# 

# %% [markdown]
# cnn.add(Dropout(0.25))  
# Dropout 레이어를 추가. 드롭아웃 비율은 0.25.  
# 학습 중에 일부 뉴런을 무작위로 비활성화하여 과적합을 방지.
# 
# cnn.add(Flatten())  
# 다차원의 특성 맵을 1차원으로 펼치는 Flatten 레이어.
# 
# cnn.add(Dense(128, activation='relu'))  
# Fully Connected 레이어를 추가. 뉴런 개수는 128개, 활성화 함수로는 ReLU.
# 
# cnn.add(Dropout(0.5))  
# Dropout 레이어는 학습 중에 각 뉴런을 유지할 확률(keep probability)을 지정.  
# 비활성화된 뉴런은 역전파 단계에서 그래디언트(gradient)를 전파시키지 않으며,  
# 이를 통해 학습 중에 다양한 뉴런들이 서로 독립적으로 학습될 수 있다.  
# 노드의 수는 Dropout 레이어를 통과한 후에도 그대로 유지.  
# **단지 학습에 활용되는 뉴런의 일부가 랜덤하게 비활성화.**
# 
# cnn.add(Dense(10, activation='softmax'))  
# Fully Connected 레이어를 추가. 출력 뉴런 개수는 10개, 활성화 함수로는 Softmax.  
# Softmax 함수는 다중 클래스 분류를 위해 확률 값을 출력.
# 

# %%
cnn.compile(loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

cnn.summary()


# %% [markdown]
# 파라미터 수 = (커널 크기 * 커널 크기 * 입력 채널 수 + 1) * 필터 수
# 
# ● 첫 번째 Conv2D 레이어:
# - 커널 크기: 3x3
# - 입력 채널 수: 1 (흑백 이미지이므로)
# - 커널 별 적용되는 bias: 1
# - 필터 수: 32
# → 파라미터 수 = (3 * 3 * 1 + 1) * 32 = 320
# 
# ● 두 번째 Conv2D 레이어:
# - 커널 크기: 3x3
# - 입력 채널 수: 32 (이전 레이어 출력 채널 수와 동일)
# - 커널 별 적용되는 bias: 1
# - 필터 수: 64
# → 파라미터 수 = (3 * 3 * 32 + 1) * 64 = 18496
# 
# ※ MaxPooling2D / Dropout / Flatten 레이어는 파라미터를 가지지 않으므로 파라미터 수는 0.
# 
# ● 첫 번째 Dense 레이어: 입력 뉴런 수: 9216 (전체 입력의 크기)/출력 뉴런 수: 128
# → 파라미터 수 = (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 = (9216 * 128) + 128 = 1179776
# 

# %% [markdown]
# 파라미터 수 = (커널 크기 * 커널 크기 * 입력 채널 수 + 1) * 필터 수
# 
# ● 첫 번째 Conv2D 레이어:
# - 커널 크기: 3x3
# - 입력 채널 수: 1 (흑백 이미지이므로)
# - 커널 별 적용되는 bias: 1
# - 필터 수: 32
# → 파라미터 수 = (3 * 3 * 1 + 1) * 32 = 320
# 
# ● 두 번째 Conv2D 레이어:
# - 커널 크기: 3x3
# - 입력 채널 수: 32 (이전 레이어 출력 채널 수와 동일)
# - 커널 별 적용되는 bias: 1
# - 필터 수: 64
# → 파라미터 수 = (3 * 3 * 32 + 1) * 64 = 18496
# 
# ※ MaxPooling2D / Dropout / Flatten 레이어는 파라미터를 가지지 않으므로 파라미터 수는 0.
# 
# ● 첫 번째 Dense 레이어: 입력 뉴런 수: 9216 (전체 입력의 크기)/출력 뉴런 수: 128
# → 파라미터 수 = (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 = (9216 * 128) + 128 = 1179776
# 

# %% [markdown]
# Output Size of Convolution  
# Convolution 연산을 수행한 후의 결과물 크기는  
# 입력 데이터의 크기, 필터 크기, 패딩의 유무, 스트라이드(stride)의 값에 의해 결정됨.
# 
# 1. 패딩을 적용하지 않은 경우:  
# 출력 크기 = (입력 크기 - 필터 크기) / 스트라이드 + 1  
# (6-2)/1 + 1 = 5 (5 * 5)
# 
# 2. 패딩을 적용한 경우:  
# 출력 크기 = (입력 크기 - 필터 크기 + 2 * 패딩) / 스트라이드 + 1  
# (6 - 2 + 2*1)/1 + 1 = 7 (7 * 7)
# 

# %% [markdown]
# ● 첫 번째 Conv2D 레이어: 입력 이미지의 크기가 28x28x1,  
# 필터의 크기는 3x3이고 총 32개의 필터를 사용. 출력 크기는  
# 출력 높이 = (입력 높이 - 필터 높이 + 2 * 패딩) / 스트라이드 + 1  
# 출력 너비 = (입력 너비 - 필터 너비 + 2 * 패딩) / 스트라이드 + 1  
# 패딩 값은 0이 되며, 스트라이드 값은 디폴트 값인 1이 적용.  
# → 출력 높이 = (28 - 3 + 0) / 1 + 1 = 26  
# → 출력 너비 = (28 - 3 + 0) / 1 + 1 = 26  
# → 출력 크기 = (None, 26, 26, 32)
# 
# ● 두 번째 Conv2D 레이어: 입력 이미지의 크기가 (None, 26, 26, 32),  
# 필터의 크기는 3x3이며 총 64개의 필터를 사용. 출력 크기는  
# → 출력 크기 = (None, 24, 24, 64)
# 
# ● MaxPooling2D 레이어가 적용되어 출력 크기는 (None, 12, 12, 64)  
# → 출력 높이 = 입력 높이 / 풀링 크기 = 24 / 2 = 12  
# → 출력 너비 = 입력 너비 / 풀링 크기 = 24 / 2 = 12
# 

# %% [markdown]
# Flatten 레이어가 적용되어 (None, 9216)의 크기로 변환. (12 * 12 * 64 = 9,216)  
# Dense 레이어의 입력 크기가 된다.
# 
# 따라서, 주어진 예제에서 9216은 Flatten 레이어 이전의 레이어에서의 출력 크기로서,  
# Dense 레이어의 입력 크기가 됩니다.
# 
# Dropout 레이어는 학습 중에 각 뉴런을 유지할 확률(keep probability)을 지정.
# 
# 비활성화된 뉴런은 역전파 단계에서 그래디언트(gradient)를 전파시키지 않으며,  
# 이를 통해 학습 중에 다양한 뉴런들이 서로 독립적으로 학습될 수 있다.
# 
# 노드의 수는 Dropout 레이어를 통과한 후에도 그대로 유지.  
# 단지 학습에 활용되는 뉴런의 일부가 랜덤하게 비활성화.
# 

# %% [markdown]
# •MaxPooling2D / Dropout / Flatten 레이어는 파라미터를 가지지 않으므로 파라미터 수는 0.
# 
# •첫 번째 Dense 레이어:  
# - 입력 크기: 9216 (Flatten 이전의 출력 크기)  
#    12 * 12 * 64  
# - 출력 크기: 128  
# - 파라미터 수 = (9216 + 1) * 128 = 1179776
# 
# •두 번째 Dropout 레이어는 파라미터를 가지지 않으므로 파라미터 수는 0.
# 
# •두 번째 Dense 레이어 (출력 레이어):  
# - 입력 크기: 128  
# - 출력 크기: 10 (클래스의 수)  
# - 파라미터 수 = (128 + 1) * 10 = 1290
# 

# %% [markdown]
# cnn.compile() : 주어진 조건으로 모델을 구성하는 모델링 역할  
# 컴파일 단계에서 모델의 손실 함수(loss function), 옵티마이저(optimizer), 그리고 평가 지표(metrics)를 설정하는 메서드.  
# 위의 코드에서는 손실 함수를 'sparse_categorical_crossentropy'를 사용하고(정수로 라벨값이 인코딩),  
# 옵티마이저로는 ‘Adam’을 선택.  
# 또한 평가 지표로는 ‘accuracy’를 설정.
# 
# cnn.summary() : 모델의 구조를 요약하여 출력하는 메서드.  
# 실행하면 모델의 레이어(layer) 정보와 각 레이어의 출력 형태(output shape) 및 파라미터 개수 등이 표시.  
# 모델의 구조를 시각적으로 파악할 수 있다.
# 

# %%
hist = cnn.fit(x_train, y_train, batch_size=128,
               epochs=30, validation_data=(x_test, y_test))


# %% [markdown]
# hist = cnn.fit(x_train, y_train, batch_size=128,  
#                epochs=30, validation_data=(x_test, y_test))  
# CNN 모델을 주어진 데이터로 학습하는 과정을 수행.
# 
# x_train은 학습 데이터의 입력, y_train은 학습 데이터의 출력(레이블).  
# batch_size=128: 한 번의 학습 단계에서 사용되는 샘플의 개수.  
# epochs=30: 전체 데이터셋을 몇 번 반복하여 학습할지.  
# validation_data=(x_test, y_test): 학습 도중에 검증을 위해 사용되는 데이터.
# 
# 학습이 진행되면서 손실 함수 값과 정확도 등의 지표가 hist 변수에 저장.  
# hist 변수는 학습 과정에서 기록된 지표들을 나타내는 정보를 담고 있다.
# 

# %%
cnn.evaluate(x_test, y_test)


# %%
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy Trend')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.grid()
plt.show()


# %%
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Trend')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.grid()
plt.show()


# %%



