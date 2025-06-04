# %%
# !pip install transformers
# !pip install tensorflow
# !pip install --upgrade --force-reinstall transformers tensorflow

# %%
import tensorflow as tf
print(tf.__version__)


# %%
from transformers import AutoTokenizer

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life",
    "I hate this so much"
]

inputs= tokenizer(raw_inputs,padding=True,truncation=True,return_tensors='tf')

# %%
print(inputs)

# %%
from transformers import AutoTokenizer

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life",
    "I hate this so much"
]
tokens = [tokenizer.tokenize(sentence) for sentence in raw_inputs]
ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]


print(ids[0])
print(ids[1])

print(tokenizer(raw_inputs,padding=True))

# %% [markdown]
# inputs는 딕셔너리 형태의 데이터로, 다양한 정보를 포함.
# "input_ids"는 토큰화된 문장의 인덱스로 이루어진 리스트.
# tokenizer.decode(inputs["input_ids"]) : 이 인덱스들을 다시 텍스트로 디코딩하여
# 원래 문장으로 복원하는 역할. 토크나이저의 역변환 과정을 보여주는 것
# 
# Embedding
# •텍스트나 단어를 수치형 벡터로 변환하는 작업
# •변환된 벡터는 기존 텍스트를 모델이 이해할 수 있는 형태로 표현
# •임베딩은 단어나 토큰 간의 의미적 유사성을 반영하기 위해 벡터 공간 내에서의
# 거리와 관계를 보존.
# • 통상 토크나이징을 통해 토큰으로 분할 후 임베딩을 통해 벡터화 진행
# 벡터화 방법: CounterVectorizer (단어의 등장 횟수를 기준으로 숫자로 표현
#  단순히 횟수만들 특징으로 잡으므로 큰 의미가 없고
#  자주 쓰이는 조사가 큰 특징으로 나타난다)
#  예) “나는 정말 열심히 살고 있다” 벡터화
#  단어사전 : 나는, 정말, 하루를 열심히 살고 있다
#  > [1, 2, 0, 1, 1, 1]
#  TfidVectorizer(Term Frequency Inverse Document Frequency Vectorizer)
#  의미가 없는 조사를 low value로 처리
# 
# Word2Vec 벡터화 방법:(단어가의 관계와 유사도를 분석)
#  위치에 나오는 단어는 비슷한 의미를 갖는다”는 분포가설에 기반
#  - CBOW:Continuos Bag of Word(주변값의 중간값으로 벡터화/예측)
# 
#  - Skip gram:중심 단어의 벡터를 보고, 주변 단어의 벡터를 예측
#  예) 예: “나는 고양이를 좋아한다”
#  중심 단어: “고양이를”
#  주변 단어: [“나는”, “좋아한다”]
# 
#  1️⃣ 새로운 문장 입력되면 초기 벡터값이 부여된다
# 예를 들어: “나는 오늘도 고양이를 좋아한다”
# 2️⃣ 이 문장의 단어들이 문맥 벡터를 만들 때
# CBOW: 주변 단어 벡터 평균으로 중심 단어 예측
# Skip-gram: 중심 단어로 주변 단어 예측
# 3️⃣ 예측 결과를 실제 단어와 비교
# 손실 함수(cross-entropy loss)가 계산됨
# 4️⃣ 손실을 최소화하기 위해 모든 단어의 벡터가 역전파로 업데이트됨.
#  (= 임베딩 테이블에 있는 각 단어의 벡터)
# 학습 데이터가 추가될 때마다
# 전체 임베딩 벡터가 조금씩 움직여서 더 나은 표현(유사 단어를 가까이
# 위치시키는 방향)으로 가게 된다
# 
# 항목 Word2Vec Transformer
# 학습 방식 CBOW/Skip-gram
# (윈도우 기반 주변 단어 예측) Self-Attention (문장 전체 관계 학습)
# 문맥 고려 주변 몇 단어(로컬 문맥)만 본다 문장 전체 (글로벌 문맥) 본다
# 추가 학습 시 주변 단어 기반의 벡터 업데이트 Self-Attention으로 모든 토큰 관계 업데이트
# 출력 벡터 고정 벡터(학습 끝나면 바뀌지 않음) 입력 문맥마다 동적 벡터 (문맥 따라 매번 다름!)
# 트랜스포머의 가중치
# • 임베딩 레이어 (토큰 벡터)
# • Self-Attention의 Query, Key, Value 가중치
# • FeedForward 네트워크 가중치
# • LayerNorm 파라미터 등
# 
# 텍스트 데이터 수집 >
# 토큰화(Tokenization) 등 전처리>
# 임베딩 학습: 토큰화된 단어들을 사용하여 단어 임베딩을 학습 >
# 임베딩 적용: 학습된 단어 임베딩을 적용하여 텍스트 데이터를 벡터로 표현 >
# 딥러닝 모델의 입력으로 사용, 단어 간의 유사도 측정, 문서 분류 등 다양한 NLP 작업
# 
# 워드 임베딩은 단어를를 고차원의 실수 벡터로 매핑 하는 과정
# 임베등 공간에서 단어의 의미와 문맥적인 유사성을 보존하기 위해 수행
# 단어 간의 의미적 유사성을 반영하면, "apple"과 "orange"는 가깝게 위치.
# 워드 임베딩이 단어의 의미를 벡터 공간 상에서 수학적인 관계로 표현할 수 있다.
# "king - man + woman = queen"과 같은 단어 간의 의미적 관계를 나타내는 벡터 연산
# 
# 주어진 문장은 단여별로 2차원의 벡터로 표현 가능(워드 임베딩 후)
# 워드 임베딩은 단어를 고차원의 실수 벡터로 매핑하는 과정
#  단어의 의미와 문맥적인 유사성을 보존하기 위해 수행
# 
#  벡터의 표현 방식: Sparse vector vs Dense vector
# sparse vector(희소 벡터)는대부분의 원소가 0인 벡터.
#  One hot encoding으로 표현된 벡터
# (원핫인코딩은 희소 벡터의 한 종류)
# Dense vector(밀집 벡터)는
# 대부분의 원소가 0이 아닌 값을 가지는 벡터.
# 즉, 벡터 내의 대다수 요소들이 0이 아닌 실수값으로 표현.
# 밀집 벡터는 단어나 개체의 의미를 더 잘 표현할 수 있으며, 유사도 계산이나 기계 학습
# 알고리즘에 더 적합
# 
# 주어진 문장의 단어를 one hot encoding표현도 가능
# 또는 Dense Vector(밀집 벡터)로 표현 가능(word Embedding)
# 
# 문장이 2개가 아니고 2천 개라면?
# 사용된 단어가 7개가 아니고 700개라면?
# 
# 1-hot encoding
# 700x700-490,000개의
# 메모리 필요
# 문장 전환 시 한 단어마다
# 700개의 숫자 필요
# 임베딩
# 2x700-1,400개의
# 메모리 필요
# 문장 전환 시 2개의 숫자
# 필요
# 
# Word2Vec은 주어진 텍스트 데이터를 사용하여 단어 간 유사성을 학습하고, 단어의 의미와 관련성을 벡터공간에서 추론할 수 있다.
# 1) gensim 라이브러리를 사용하여 Word2Vec 모델을 학습.
# 2) 텍스트 데이터를 sentences라는 단어의 시퀀스로 변환한 후, Word2Vec 모델 객체를 생성.
# 모델의 파라미터로는 벡터의 크기(size), 윈도우 크기(window size), 최소 등장
# 횟수(min_count) 등을 설정.
# 3) 학습된 모델에서는 단어의 벡터 표현을 확인할 수 있다.
# 
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
# 각 단어를 100차원의 벡터로 표현하며, 각 단어는 앞뒤로 5개의 단어를 고려하여 학습되며, 각 단어는 적어도 한 번 이상 나타나는 모든 단어를 학습하는 Word2Vec 모델을 생성
# 
# Word2Vec 모델을 생성하는 코드:
# sentences: 학습할 문장의 리스트.
# 
# vector_size: 생성될 각 단어 벡터의 차원. 클수록 좋지만, 메모리 사용량과 학습 시간에 영향.
# 
# window: 단어가 고려해야 하는 컨텍스트의 크기.
# window 값이 5라면, 각 단어는 앞 뒤로 5개의 단어를 고려하여 학습.
# 
# min_count: 모델에 학습할 때 고려할 단어의 최소 빈도수.
# 이 값이 1이라면, 각 단어가 적어도 한 번은 나타나야 한다.
# vector = model.wv['NLP']
# Word2Vec 모델에서 ‘NLP’라는 단어에 해당하는 벡터를 가져와 vector라는 변수에 저장
# 
# model: 단어를 벡터로 변환.
# 
# wv: Word2Vec 모델에서 단어 벡터를 가져오는 방법
#       wv는 “word vector”, 특정 단어에 대한 벡터를 조회하는 데 사용.
# 
# ‘NLP’: ‘NLP’라는 단어에 대한 벡터를 가져온다.
# 

# %%
# !pip install gensim
# !pip install --upgrade numpy gensim
# !pip install nltk

# %%
from gensim.models import Word2Vec

# 텍스트 데이터를 전처리 하여 단어의 시퀀스로 변환
sentences = [['I','love','NLP'],['Word2Vec','is','awesome']]

# Word2Vec 모델 학습
model =  Word2Vec(sentences,vector_size=100,window=5,min_count=1)

# 단어의 벡터 표현 확인
vector = model.wv['NLP']
print(vector)

# %%
import nltk
nltk.download('punkt_tab')

# %%
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# 문장 데이터
sentences = [
 'I love natural language processing',
 'Word2Vec is a popular word embedding model',
 'Natural language processing is an important field in Al'
]

# 문장을 토큰화
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
print(tokenized_sentences)

# %%
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

MY_LENGTH = 80
MY_WORD = 5000
MY_SAMPLE = 10

# %%
from keras.datasets import imdb

# IMDB 데이터셋 로드
(x_train, y_train),(x_test, y_test) = imdb.load_data()

# 전체 데이터셋을 합친 후 단어의 종류의 수 확인(약 88,585 종류)
word_set = set()
for review in x_train + x_test:
    word_set.update(review)

num_words=len(word_set)
print("단어 종류의 수:", num_words)

# %%
(X_train, Y_train),(X_test, Y_test) =  imdb.load_data(num_words =  MY_WORD)
# 자주 등장하는 상위 My_WORD개의 단어를 지정하여 업로드
# IMDB word index
print('샘플 영화평 \n',X_train[MY_SAMPLE]) # 지정된 샘플(10)의 데이터(정수)를 보여준다
print('총 단어 수: \n',len(X_train[MY_SAMPLE])) # 지정된 샘플의 단어 갯수
print('감성(0=부전, 1=긍정): \n', Y_train[MY_SAMPLE]) # 지정된 샘플의 라벨 표시

# %% [markdown]
# 전체 데이터 셋을 합친 단어의 종류의 수는 약 88,585 종류
# 그 중 5000개의 단어만 업로드 하여 x,y데이터로 나눔
# 샘플 데이터 5번의 영화평을 출력 : 정수로 처리된 단어의 인덱스, 단어 수,감성 라벨

# %%
from tensorflow.keras.datasets import  imdb

MY_SAMPLE = 10 #사전 단어 수
MY_WORD = 5000 # 샘플 영화평

# 데이터 로드
(x_train,_),(_,_) = imdb.load_data(num_words=MY_WORD)

# word-index 매핑 가져오기
word_index = imdb.get_word_index()

# word-index 매핑 반전하여 index to word 매핑 생성
index_to_word = {index: word for word, index in word_index.items()}

# 샘플 영화평을 단어로 변환하여 출력
decoded_review = ' '.join(index_to_word[index] for index in x_train[MY_SAMPLE])
print('샘플 영화 평: \n', decoded_review)

# %% [markdown]
# 🟦 IMDB 데이터셋 로드 및 단어 복원 흐름 설명
# imdb.load_data(num_words=MY_WORD)
#  → IMDB 데이터셋을 로드할 때 사용할 단어 수(MY_WORD)로 제한함.
#   예: num_words=5000이면 상위 5000개의 단어만 사용.
# 
# (x_train, _), (_, _) = imdb.load_data(num_words=MY_WORD)
#  → 훈련용 리뷰 데이터를 x_train에, 라벨은 _로 무시하고 저장함.
# 
# imdb.get_word_index()
#  → 단어를 키, 정수 인덱스를 값으로 갖는 딕셔너리 형태의 word_index 반환.
#   예: 'fawn': 34701, 'tsukino': 52006 ...
# 
# index_to_word = {index: word for word, index in word_index.items()}
#  → 인덱스-단어 구조로 word_index를 반전시켜 index_to_word 딕셔너리 생성.
#   즉, 정수 인덱스를 통해 단어를 다시 찾을 수 있게 함.
# 
# ' '.join(index_to_word[index] for index in x_train[MY_SAMPLE])
#  → x_train[MY_SAMPLE]에 있는 단어 인덱스들을 실제 단어로 바꾸고 공백으로 연결.
#   하나의 문장(리뷰)으로 복원하여 출력 가능.
#   예: [1, 14, 20, 6] → "the movie was good"

# %%
print(word_index)

# %%
print(index_to_word)

# %%
top_words = [word for word, index in word_index.items() if index <= MY_WORD]

print('상위 {}개의 단어 목록:'.format(MY_WORD))
for i, word in enumerate(top_words):
    print(i+1,word)

# %%
# 각 영화평의 길이를 일정하게 맞춤

X_train = pad_sequences(sequences=X_train,truncating='post', #뒷부분 삭제,pre는 앞부분
padding ='post', maxlen = MY_LENGTH) # 80단어보다 짧으면 뒤(post) 를 0으로 채움

X_test = pad_sequences(sequences=X_test,truncating='post',
padding = 'post', maxlen=MY_LENGTH)

print('\n학습용 입력 데이터 모양: ',X_train.shape)
print('학습용 출력 데이터 모양: ',Y_train.shape)
print('평가용 입력 데이터 모양: ',X_test.shape)
print('평가용 출력 데이터 모양: ',Y_test.shape)
        

# %% [markdown]
# 다음은 이미지에서 추출한 내용을 **설명 댓글** 형식으로 깔끔하게 정리한 것입니다:
# 
# ---
# 
# ### 🟦 시퀀스 데이터 패딩: `pad_sequences` 사용 설명
# 
# * `X_test = pad_sequences(sequences=X_test, truncating='post', padding='post', maxlen=MY_LENGTH)`
#    → 시퀀스 데이터를 일정한 길이로 맞추기 위해 **패딩(padding)** 또는 **절단(truncating)** 작업을 수행한다.
# 
# ---
# 
# #### 🔹 padding:
# 
# * 시퀀스의 길이를 `maxlen`에 맞추기 위해 **뒤쪽에 0을 추가**한다.
# * `'post'` 설정 시, 시퀀스의 **뒷부분에 패딩**을 추가한다.
#    예: `[1, 2, 3]` → `[1, 2, 3, 0, 0]`
# 
# #### 🔹 truncating:
# 
# * 시퀀스가 너무 길 경우 **뒷부분을 잘라낸다**.
# * `'post'` 설정 시, 시퀀스의 **뒤쪽을 잘라낸다**.
#    예: `[1, 2, 3, 4, 5, 6]` → `[1, 2, 3, 4, 5]` (if `maxlen=5`)
# 
# ---
# 
# ### ✅ 요약
# 
# * `padding='post'`: 시퀀스 뒤에 패딩 추가
# * `truncating='post'`: 시퀀스 뒤를 잘라냄
# * `maxlen=...`: 최종 시퀀스 길이 지정
# 
# ---
# 
# 필요하시면 `pad_sequences`의 앞쪽 패딩 `'pre'`과 비교하거나 실제 데이터 전처리 코드도 제공해 드릴 수 있습니다!
# 

# %%
# RNN 구현
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Embedding(input_dim=MY_WORD, output_dim=128,
                    input_length=MY_LENGTH))
model.add(Bidirectional(LSTM(units=64)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(LSTM(units=64, input_shape=(MY_LENGTH, 32)))
model.add(Dense(units=1, activation='sigmoid'))

print('RNN 요약')
model.build(input_shape=(None, MY_LENGTH))
model.summary()



# %% [markdown]
# 임베딩 차원 확장: 단어 의미를 더 잘 표현
# 양방향 LSTM: 문맥을 앞뒤로 살펴서 더 풍부하게 학습
# LSTM 층 쌓기: 깊은 네트워크로 성능 향상
# Dropout: 과적합 방지
# RMSprop: LSTM에 자주 쓰이는 옵티마이저
# validation_split: 학습 중 과적합 여부 확인
# 추가 지표(confusion matrix, classification report): 모델의 강점과 약점 파악
# 시각화: 학습과정 파악

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

model = Sequential()
model.add(Embedding(input_dim=MY_WORD, output_dim=128, input_length=MY_LENGTH))

# 양방향 LSTM → 시퀀스 출력 유지
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))

# Stacked LSTM (계속 시퀀스 유지)
model.add(LSTM(64, return_sequences=True))

# 마지막 LSTM → 시퀀스 X, 최종 벡터만 출력
model.add(LSTM(64))

# Dropout 적용
model.add(Dropout(0.5))

# 출력층
model.add(Dense(units=1, activation='sigmoid'))

# 모델 구조 확인
print('RNN 요약')
model.build(input_shape=(None, MY_LENGTH))
model.summary()


# %% [markdown]
# 다음은 두 이미지에서 추출한 **텍스트 전체 내용**입니다:
# 
# ---
# 
# ## ✅ 이미지 1 텍스트
# 
# ```python
# model.add(Embedding(input_dim=MY_WORD, output_dim=32, input_length=MY_LENGTH))
# ```
# 
# * model.add() 메서드를 사용하여 그 안에 Embedding 레이어를 추가.
# * Embedding은 단어를 고정된 크기의 벡터로 변환해주는 층.
# * input\_dim: 모델이 사용할 전체 단어의 개수. 전체 단어 집합의 크기.
# * output\_dim: 단어 임베딩 차원 수. 임베딩 벡터의 차원
# * input\_length: 입력 시퀀스의 길이 설정.
# 
# \*Embedding: 단어에 벡터를 할당 표현하여 고정된 크기로 추후 사용.
# 
# * 예) 단어 수가 많은 텍스트 데이터를 숫자로 바꿔야 하므로 임베딩 층을 사용.
# * 입력 정수 → 임베딩 벡터로 매핑
# * Embedding은 모델의 입력층 바로 다음에 위치하며, 훈련 과정에서 학습된다.
# * 텍스트 전처리 과정에서 수치화된 데이터를 임베딩 층에 통과시켜 단어 벡터 생성
# * 입력 정수는 단어의 인덱스이며, 각 인덱스는 고유한 임베딩 벡터에 매핑된다.
# * Embedding층은 학습을 통해 각 단어의 의미를 파악하여 벡터를 구성
# 
# ```python
# model.add(LSTM(units=64, input_shape=(MY_LENGTH, 32)))
# ```
# 
# * LSTM은 장기 의존성을 학습할 수 있는 순환 신경망 레이어.
# * \*units는 LSTM 레이어의 출력 차원 또는 노드의 개수
# * \*input\_shape는 입력 시퀀스의 형태. (시퀀스 길이, 입력 차원)
# 
# ---
# 
# ## ✅ 이미지 2 텍스트
# 
# ### 파라미터 계산 공식
# 
# **1. Embedding 레이어:**
# 
# * 파라미터 개수 = 입력 차원 × 출력 차원
#   예) 입력 차원: 단어 수(5000), 출력 차원: 임베딩 크기(32)
#   → 5000 × 32 = **160,000**
# 
# **2. LSTM 레이어:**
# 
# * 파라미터 개수 = (입력 차원 + LSTM 내부 상태 크기) × 4 × LSTM 내부 상태 크기
#   예) 입력 차원 32, LSTM 상태 크기 64
#   → (32 + 64) × 4 × 64 = **24,576**
# 
# **3. Dense 레이어:**
# 
# * 파라미터 개수 = (입력 차원 + 편향) × 출력 차원
#   예) 입력 차원 64, 출력 차원 1
#   → (64 + 1) × 1 = **65**
# 
# ---
# 
# ### LSTM 레이어는
# 
# 게이트(gate)라고 불리는 구조를 가지고 있으며,
# 게이트의 역할은 어떤 정보를 기억하고 어떤 정보를 전달할지를 결정하는 것.
# LSTM의 파라미터는 입력 게이트, 망각 게이트, 출력 게이트, 셀 상태를 조절
# 
# LSTM 레이어의 파라미터 개수 계산식:
# 
# * 입력 차원(input\_dim): 이전 층의 출력 차원 등
# * LSTM 내부 상태 크기(units): LSTM 레이어가 가지는 메모리 셀의 개수를 의미.
# 
# > 예: input\_dim=32, units=64인 경우
# > → (32 + 64) × 4 × 64 = **24,576**
# 
# * 왜 ×4?: LSTM은 4개의 게이트를 학습하기 때문 (입력 게이트, 망각 게이트, 출력 게이트, 셀 상태 게이트)
# * LSTM은 gate마다 weight를 가지므로, 총 파라미터 수는 4배로 계산됨.
# 
# ---
# 
# 필요하시면 이 내용을 슬라이드 형태나 요약표로도 정리해드릴 수 있어요!
# 

# %%
# RNN 학습
# model.compile(optimizer='adam', loss='binary_crossentropy',
# metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',     # 모니터할 기준 (val_accuracy 도 가능)
    patience=10,             # 개선이 없더라도 기다릴 에포크 수
    restore_best_weights=True  # 가장 성능 좋았던 가중치로 복원
)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print('\n 학습 시작')
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=200,
    validation_split=0.2,
    callbacks=[early_stop]
)

# %%
# RNN 평가
score = model.evaluate(x=X_test, y=Y_test,verbose=1)
print("최종 정확도: {:.2f}".format(score[1]))

# %%
# RNN 예측

test = X_test[MY_SAMPLE].reshape(1,80)
pred = model.predict(test)
pred = (pred>0.5)
from sklearn.metrics import classification_report, confusion_matrix
pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

print('\n 샘플 영화평 :\n', test)
print('RNN 감성 예측 :', pred)
print('정담(0=부정,1=긍정):', Y_test[MY_SAMPLE])

# %%
import matplotlib.pyplot as plt

# history = model.fit(...)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('모델 정확도')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()



