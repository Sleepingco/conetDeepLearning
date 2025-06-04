# %% [markdown]
# 처리 단계 처 리 내 용 요 약
# 데이터 준비 BBC 데이터셋을 불러와서 기사 내용과 레이블을 추출.
# 데이터 전처리 불용어를 제거 후 기사 내용을 토큰화하고, 단어를 숫자로 인코딩하여 벡터로 변환.
# 기사 내용의 길이를 맞추기 위해 패딩을 추가.
# 레이블을 원-핫 인코딩하여 벡터로 변환.
# 데이터 분할 전체 데이터를 학습 데이터와 테스트 데이터로로.
# 모델 구성 RNN(Recurrent Neural Network) 모델을 구성
# 임베딩 레이어를 추가하여 단어 벡터를 입력으로 받는다.
# RNN 층을 추가하여 시퀀스 데이터를 처리
# 출력 층을 추가하여 분류 결과를 예측
# 모델 컴파일 손실 함수, 최적화 알고리즘, 평가 지표를 설정.
# 모델 학습 학습 데이터를 사용하여 모델을 학습
# 모델 평가 테스트 데이터를 사용하여 모델의 성능을 평가.
# 정확도, F1 스코어 등을 계산하여 분류 성능을 확인.
# 모델 예측 새로운 기사를 입력으로 받아 모델을 사용하여 분류 결과를 예측
# 
# NLTK(Natural Language Toolkit)자연어 처리를 위한 파이썬 라이브러리
# 영어 불용어(stopwords)
# 자연어 처리 작업에서 의미가 없거나 분석에 영향이 크지 않는 일반적인 단어들
# 텍스트 데이터를 전처리 할 때 사용.
# 주로 문법적인 역할을 하는 단어들, 텍스트 분석에 자주 등장하지만 중요한 의미를
# 전달하지 않는 경우가 많다.
# 불용어를 제거함으로써 모델의 성능을 향상시킬 수 있다.
# 예) "a", "an", "the", "is", "of", "and" 등 문장의 문법적인 구조를 형성하는 데 도움을
# 주지만, 분석에는 큰 영향을 미치지 않는다.
# nltk.corpus.stopwords 모듈을 통해 영어 불용어를 사용할 수 있다. 

# %%
import csv
import numpy as np
import nltk
from time import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM,Embedding

# %%
MY_VOCAB = 5000 # 학습용으로 사용할 단어의 전체 갯수
MY_EMBED = 64 # 벡터화할 임베딩 규모(차원)
MY_HIDDEN = 100 # RNN 셀의 규모(숨겨진 레이어층)
MY_LEN = 200 # 업로드할 기사의 전체 단어의 숫자

MY_SPLIT = 0.8 # 학습용 데이터 비율
MY_SAMPLE = 123 # 샘플로 사용할 기사 번호
MY_EPOCH = 100 # 반복 학습수
original = []
processed = []
labels = []

# %%
# NLTK(Natural Language Toolkit)에서 영어의 불용어(stop words)를 다운로드
nltk.download('stopwords') # 기사 분류에 의미없는 영어 단어 다운로드
MY_STOP = set(nltk.corpus.stopwords.words('english'))

# corpus : 텍스트 데이터를 분석/처리 위한 여러 문서나 말뭉치들

print('영어 제외어',MY_STOP)
print('제외어 갯수', len(MY_STOP))
print(type(MY_STOP))
print('the' in MY_STOP)

# %%
import os
print("현재 작업 디렉토리:", os.getcwd())


# %%
import csv

# 파일 경로 설정 (Windows에서는 이스케이프 처리 필요)
path = r'.\bbc-text.csv'


# 리스트 초기화
labels = []
original = []
processed = []

# CSV 파일 열기
with open(path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # 헤더 읽기
    print("헤더:", header)

    # 내용 확인 (옵션)
    for row in reader:
        # print(row)
        labels.append(row[0])      # 예: 카테고리이름을 리스트에 추가
        original.append(row[1])    # 예: 뉴스 본문 내용을 리스트에 추가
        news = row[1] # 각행의 두 번째 열값을 news 변수에 할당

        # print('before:',news)#제외어 처리 전 문장
        for word in MY_STOP:
            mask = ' ' + word+ ' '
            news = news.replace(mask,' ')
        # print('after :', news) # 제외어 처리 후 문장
        processed.append(news) # 처리된 결과인 news를 processed 리스트에 추가
    print('제외어 처리 전체 기사 수: ',len(processed))
    print('레이블 데이터 수:', len(labels))
    print('Original 데이터 수:', len(original))
    print('News 데이터 수:',len(news))

# %%


## ✅ 코드 내용 정리

```python
for row in reader:
    labels.append(row[0])       # 각 행의 카테고리 이름을 labels 리스트에 추가
    original.append(row[1])     # 각 행의 기사 내용을 original 리스트에 추가
    news = row[1]               # 각 행의 두 번째 열 값을 news 변수에 할당
```

👉 **→ `labels`, `original` 리스트에 읽은 기사를 한 줄씩 추가 후 `news`의 내용은 별도 저장**

---

```python
for word in MY_STOP:
    mask = ' ' + word + ' '
    news = news.replace(mask, ' ')
    processed.append(news)
```

👉 **→ `news`의 내용에서 불용어를 찾아서 양쪽 끝에 공백을 넣어 `mask`로 담은 후,
해당 `mask`를 공백으로 대체해 `news`에 저장하고,
불용어가 사라진 `news`의 내용을 `processed`에 저장해 놓는다.**

---

## ✅ 보충 설명

> `"and"`라는 불용어 제거 시,
> `"sand"`라는 단어까지 `"s "`로 바뀌어버리는 문제를 해결한다.

* 즉, `mask = ' ' + word + ' '` 로 공백을 넣은 이유는,
  `" and "`만 제거하고 `"sand"`는 건드리지 않기 위함이다.

---

## ✅ 핵심 요약

| 리스트         | 역할                        |
| ----------- | ------------------------- |
| `labels`    | 뉴스의 카테고리(예: 스포츠, 정치 등) 저장 |
| `original`  | 뉴스 본문 원문 저장               |
| `processed` | 불용어 제거된 뉴스 본문 저장          |

| 전략                        | 이유                          |
| ------------------------- | --------------------------- |
| `' ' + word + ' '` 사용     | 정확한 단어 매칭만 제거 (단어 일부 제거 방지) |
| `news.replace(mask, ' ')` | 공백으로 치환하여 단어 위치 유지          |

---


# %%
print('샘플 기사 원문 :',original[MY_SAMPLE])
print('샘플 기사 분류 :',labels[MY_SAMPLE])
print('샘플 기사 기사 데이터:',type(original[MY_SAMPLE]))
print('샘플 기사의 젠체 단어수:',str(len(original[MY_SAMPLE])).split())

print('\n제외어를 삭제한 샘플 기사:',processed[MY_SAMPLE])
print('제외어 삭제 샘플의 단어수:',str(len(processed[MY_SAMPLE])).split())
# 정수에 split() 함수를 적용할 수 없기 때문에
# 문자열을 분할하기 위해 split()함수를 문자열에 적용해아

# %%
# 텍스트 데이터를 작은 단위로 분할하는 처리(객체생성>토큰화)
A_token = Tokenizer(num_words= MY_VOCAB,
oov_token='!') # out of vocaburary > "!"
A_token.fit_on_texts(processed)

print('총 기사 수:',A_token.document_count)
print('총 단어 수:',len(A_token.word_counts))
print('각 단어의 사용 횟수:',A_token.word_counts)
print('단어를 정수로:',A_token.word_index)

# %% [markdown]
# 
# 
# ## ✅ 코드 내용
# 
# ```python
# A_token = Tokenizer(num_words=MY_VOCAB, oov_token='!')
# ```
# 
# ### 🔹 인자 설명
# 
# * **`num_words`**:
#   토큰화할 단어의 최대 개수를 지정.
#   → 가장 빈도가 높은 상위 `num_words`개의 단어만 토큰화에 사용.
# 
# * **`oov_token='!'`**:
#   사전에 없는 단어(OOV: Out-Of-Vocabulary)를 나타내는 토큰을 `'!'`로 지정.
#   → 학습하지 않은 단어가 입력되면 `'!'`로 대체됨.
# 
# ---
# 
# ```python
# A_token.fit_on_texts(processed)
# ```
# 
# ### 🔹 동작 설명
# 
# * **불용어가 제거된 텍스트 리스트 `processed`** 를 입력으로 받아
# * 각 단어에 고유한 **정수 인덱스**를 부여하며 내부 단어 사전을 구축함.
# 
# ---
# 
# ## ✅ 핵심 요약
# 
# | 용어               | 설명                               |
# | ---------------- | -------------------------------- |
# | `Tokenizer`      | 텍스트를 숫자 인덱스로 변환해주는 Keras 도구      |
# | `fit_on_texts()` | 전체 문서로부터 단어의 빈도수를 학습해 정수 인덱스를 매핑 |
# | `oov_token`      | 훈련 시 사전에 없는 단어를 처리할 수 있도록 설정     |
# | 결과               | 각 단어에 대해 정수 인덱스가 부여된 단어 사전 생성됨   |
# 
# ---
# 

# %%
# 입력된 텍스트 데이터를 정수 시퀀스로 변환
A_tokenized = A_token.texts_to_sequences(processed)

print('토큰 처리된 데이터 type:',type(A_tokenized))
print('토큰 처리된 데이터 type:',len(A_tokenized))
print('토큰 처리된 데이터 type:',A_tokenized[MY_SAMPLE])

# %% [markdown]
# 불용어 처리 데이터를 입력으로 받아서 각각 정수 시퀀스로 변환.
# 이전 단계에서 생성한 Tokenizer 객체의 단어 사전을 활용하여 각 단어를 정수로 매핑
# 텍스트 데이터를 수치형 데이터로 변환하면, 기사 데이터를 신경망 모델에 입력으로
# 사용할 수 있다.

# %%
longest = max([len(x) for x in A_tokenized])
print('제일 긴 기사 단어 수 :',longest)
shortest = min([len(x) for x in A_tokenized])
print('제일 짧은 기사 단어 수 :',shortest)

# %%
A_tokenized = pad_sequences(A_tokenized,
maxlen = MY_LEN,
padding='pre',
truncating='pre')
print('샘플 기사 길이 처리본\n',A_tokenized[MY_SAMPLE])
print('토큰 처리된 데이터 수;',len(A_tokenized))

# %% [markdown]
# maxlen 파라미터 (200)를 사용하여 패딩 후 시퀀스의 최대 길이를 지정하고,
# padding 파라미터 (pre)와 truncating 파라미터 (pre)를 사용하여 패딩의 위치를 지정

# %%
longest = max([len(x) for x in A_tokenized])
print('제일 긴 기사 단어 수 :',longest)
shortest = min([len(x) for x in A_tokenized])
print('제일 짧은 기사 단어 수 :',shortest)

# %%
C_token = Tokenizer()
C_token.fit_on_texts(labels)

print('총 기사 수:',C_token.document_count)
print('총 단어 수:',len(C_token.word_counts))
print('각 단어의 사용 횟수:',C_token.word_counts)
print('단어를 정수로:',C_token.word_index)

# %%
C_tokenized = C_token.texts_to_sequences(labels)
C_tokenized = np.array(C_tokenized)

print(C_tokenized)

# %% [markdown]
# 1. 텍스트 데이터 준비:
# text_data = ["This is an example sentence", "Another example sentence"]
# 2.Tokenizer 객체 생성:
# tokenizer = Tokenizer()
# 3.텍스트 데이터 입력하여 단어 사전 구축 및 인덱스
# tokenizer.fit_on_texts(text_data)
# 4.텍스트 데이터를 정수 시퀀스로 변환:
# tokenizer.texts_to_sequences(text_data)
# 5.패딩 또는 자르기 수행:
# pad_sequences(sequences, maxlen=max_length, padding='post')

# %%
from sklearn.model_selection import train_test_split

X_train, X_test,Y_train,Y_test = train_test_split(A_tokenized,C_tokenized,train_size = MY_SPLIT,shuffle=False)
Y_train = np.array(Y_train) - 1
Y_test = np.array(Y_test) - 1
print('학습용 입력 데이터 모양:',X_train.shape)
print('학습용 출력 데이터 모양:',Y_train.shape)
print('단어를 정수로:',C_token.word_index)

# %%
model = Sequential()
model.add(Embedding(input_dim=MY_VOCAB,output_dim=MY_EMBED))
model.add(Dropout(rate=0.5))
model.add(LSTM(units=MY_HIDDEN))
model.add(Dense(units=5,activation='softmax'))

# 데이터 한 번 넣어서 강제로 빌드
model.build(input_shape=(None, MY_LEN)) # None은 배치크기
print('RNN 요약')
print("=========")
model.summary()

# %%
model = Sequential()
model.add(Embedding(input_dim=MY_VOCAB,output_dim=MY_EMBED))
model.add(Dropout(rate=0.5))
model.add(LSTM(units=MY_HIDDEN))
model.add(Dense(units=5,activation='softmax'))

# 데이터 한 번 넣어서 강제로 빌드
model.build(input_shape=(None, MY_LEN)) # None은 배치크기
print('RNN 요약')
print("=========")
model.summary()

# %%
model = Sequential()
model.add(Embedding(input_dim=MY_VOCAB, output_dim=MY_EMBED))
model.add(Bidirectional(LSTM(units=MY_HIDDEN, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5, activation='softmax'))
# 데이터 한 번 넣어서 강제로 빌드
model.build(input_shape=(None, MY_LEN)) # None은 배치크기
print('RNN 요약')
print("=========")
model.summary()

# %%
model.compile(optimizer='adam',
loss ='sparse_categorical_crossentropy',
metrics=['acc'])
print('학습 시작')

# %%
begin = time()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',     # 모니터할 기준 (val_accuracy 도 가능)
    patience=20,             # 개선이 없더라도 기다릴 에포크 수
    restore_best_weights=True  # 가장 성능 좋았던 가중치로 복원
)

model.fit(X_train,Y_train,epochs=MY_EPOCH,verbose=1,batch_size=64,validation_split=0.2,callbacks=[early_stop])
end = time()
print('training time : {:.2f}초'.format(end-begin))

# %%
import numpy as np
print(np.unique(X_train))
print(np.unique(Y_train))


# %%
score = model.evaluate(X_test, Y_test,verbose=1)
print('최종 정확도: {:.2f}'.format(score[1]))

# %%
pred = model.predict(X_test)
print('추측값\n',pred)
pred = pred.argmax(axis=1)
print('추측값(argmax 처리 후)\n',pred)
print('정답\n', Y_test.flatten())

# %%
correct_predictions = 0
total_predictions = len(Y_test)
for i in range(total_predictions):
    if pred[i] == Y_test[i]:
        correct_predictions +=1
accuracy = correct_predictions/total_predictions
print('정확도', accuracy)

# %%

news = ['Bayern Munich cruised to an emphatic 7-0 victory over VfL Bochum \
 5 to move to the top of the German Bundesliga.']

news = A_token.texts_to_sequences(news)
print(news)
print('총 단어 수 :',len(news[0]))

# %%
news = pad_sequences(news, maxlen=MY_LEN,padding='pre',truncating='pre')
print('총단어수 : ', len(news[0]))

pred = model.predict(news)
pred = pred.argmax(axis=1)
print('RNN 추측값 : ',pred)

# %%



