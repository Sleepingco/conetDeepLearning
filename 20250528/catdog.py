# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import zipfile

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test1")

# %%
print(os.listdir("/kaggle/working/test1"))
print(os.listdir("/kaggle/working/train"))

# %%
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

# %%
# filenames = os.listdir("/kaggle/working/train/train")
# categories = []
# for filename in filenames:
#     category = filename.split('.')[0]
#     if category == 'dog':
#         categories.append(1)
#     else:
#         categories.append(0)

# df = pd.DataFrame({
#     'filename': filenames,
#     'category': categories
# })

# %%
import os
import pandas as pd

filenames = os.listdir("/kaggle/working/train/train")
categories = []

for filename in filenames:
    label = filename.split('.')[0]
    if label == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# 🔥 문자열로 바꿔주기 (중요)
df["category"] = df["category"].astype(str)


# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="/kaggle/working/train/train",
    x_col="filename",
    y_col="category",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="/kaggle/working/train/train",
    x_col="filename",
    y_col="category",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)


# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df['category'].value_counts().plot.bar()
print(df['category'].value_counts())

# %%
# 예: 파일 이름 리스트 중 무작위 선택
sample = random.choice(filenames)

# 경로 생성 (슬래시가 누락되지 않도록 os.path.join 사용 권장)
image_path = os.path.join("/kaggle/working/train/train", sample)

# 이미지 로드
image = load_img(image_path)

# 이미지 시각화
plt.imshow(image)
plt.axis('off')  # 축 숨기기 (선택)
plt.show()

# %% [markdown]
# From our data we have 12000 cats and 12000 dogs

# %% [markdown]
# # Build Model
# 
# <img src="https://i.imgur.com/ebkMGGu.jpg" width="100%"/>

# %% [markdown]
# * **Input Layer**: It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
# * **Conv Layer**: This layer will extract features from image.
# * **Pooling Layer**: This layerreduce the spatial volume of input image after convolution.
# * **Fully Connected Layer**: It connect the network from a layer to another layer
# * **Output Layer**: It is the predicted values layer. 

# %%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# %% [markdown]
# # Callbacks

# %%
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# %% [markdown]
# **Early Stop**
# 
# To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased

# %%
earlystop = EarlyStopping(patience=10)

# %% [markdown]
# **Learning Rate Reduction**
# 
# We will reduce the learning rate when then accuracy not increase for 2 steps

# %%
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# %%
callbacks = [earlystop, learning_rate_reduction]

# %% [markdown]
# # Prepare data

# %% [markdown]
# Because we will use image genaretor `with class_mode="categorical"`. We need to convert column category into string. Then imagenerator will convert it one-hot encoding which is good for our classification. 
# 
# So we will convert 1 to dog and 0 to cat

# %%
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

# %%
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# %%
train_df['category'].value_counts().plot.bar()

# %%
validate_df['category'].value_counts().plot.bar()
print(validate_df['category'].value_counts())

# %%
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

# %% [markdown]
# # Traning Generator

# %%
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/working/train/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# %% [markdown]
# ### Validation Generator

# %%
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/kaggle/working/train/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# %% [markdown]
# # See how our generator work

# %%
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/kaggle/working/train/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

# %%
import matplotlib.pyplot as plt

# figure 생성
plt.figure(figsize=(12, 12))

# generator에서 15장의 이미지만 수집
count = 0
for X_batch, Y_batch in example_generator:
    for i in range(X_batch.shape[0]):
        if count >= 15:
            break
        plt.subplot(5, 3, count + 1)
        plt.imshow(X_batch[i])
        plt.axis('off')  # 축 제거
        count += 1
    if count >= 15:
        break

plt.tight_layout()
plt.show()


# %% [markdown]
# Seem to be nice 

# %% [markdown]
# # Fit Model

# %%
epochs=3 if FAST_RUN else 50
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

# %% [markdown]
# # Save Model

# %%
model.save_weights("model.weights.h5")

# %% [markdown]
# # Virtualize Training

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Prepare Testing Data

# %%
test_filenames = os.listdir("/kaggle/working/test1/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

# %%
print(test_df['category'].value_counts())

# %% [markdown]
# # Create Testing Generator

# %%
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/kaggle/working/test1/test1", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

# %% [markdown]
# # Predict

# %%
steps = int(np.ceil(nb_samples / batch_size))
predict = model.predict(test_generator, steps=steps)


# %% [markdown]
# For categoral classication the prediction will come with probability of each category. So we will pick the category that have the highest probability with numpy average max

# %%
test_df['category'] = np.argmax(predict, axis=-1)

# %% [markdown]
# We will convert the predict category back into our generator classes by using `train_generator.class_indices`. It is the classes that image generator map while converting data into computer vision

# %%
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

# %% [markdown]
# From our prepare data part. We map data with `{1: 'dog', 0: 'cat'}`. Now we will map the result back to dog is 1 and cat is 0

# %%
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

# %% [markdown]
# ### Virtaulize Result

# %%
test_df['category'].value_counts().plot.bar()

# %% [markdown]
# ### See predicted result with images

# %%
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("/kaggle/working/test1/test1/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()

# %%
predict = model.predict(test_generator, steps=int(np.ceil(nb_samples / batch_size)))


# %%
# 1. 예측된 숫자 클래스 저장
test_df['category'] = np.argmax(predict, axis=-1)

# 2. 숫자 → 문자열로 변환해 별도 컬럼 생성
test_df['label_str'] = test_df['category'].map({0: 'cat', 1: 'dog'})
 # ✅ 여기까지만


# %%
print(test_df[['filename', 'category', 'label_str']].head())


# %%
test_df['category'] = np.argmax(predict, axis=-1)

# 강제 변환을 위해 int 타입으로 명시
test_df['category'] = test_df['category'].astype(int)

# 명시적 매핑
test_df['label_str'] = test_df['category'].map({0: 'cat', 1: 'dog'})
print(test_df[['filename', 'category', 'label_str']].head())


# %%
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

# 1. 예측 결과 처리
# 숫자 예측 결과
test_df['category'] = np.argmax(predict, axis=-1)

# 강제 변환을 위해 int 타입으로 명시
test_df['category'] = test_df['category'].astype(int)

# 명시적 매핑
test_df['label_str'] = test_df['category'].map({0: 'cat', 1: 'dog'})

# 2. 샘플 18장만 시각화
sample_test = test_df.head(18)

# 3. 이미지 시각화
plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():
    filename = row['filename']
    label_str = row['label_str']  # ✅ 이 줄을 주석 해제해야 함

    img_path = f"/kaggle/working/test1/test1/{filename}"
    img = load_img(img_path, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.title(f"{filename}", fontsize=10)
    plt.xlabel(f"Prediction: {label_str}", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Model Predictions", fontsize=16)
plt.show()


# %% [markdown]
# # Submission

# %%
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)

# %%
import matplotlib.pyplot as plt

# 훈련 기록
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# 1. 정확도 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, '-', label='Training Accuracy')       # '-' → 선만
plt.plot(epochs, val_acc, '--', label='Validation Accuracy') # '--' → 점선
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 2. 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, '-', label='Training Loss')
plt.plot(epochs, val_loss, '--', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



