import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('hepsiburada.csv')

target = dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()

cutoff = int(len(data) * 0.80)
x_train, x_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

model = Sequential()
embedding_size = 50
model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_tokens, name='embedding_layer'))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

model.fit(x_train_pad, y_train, epochs=5, batch_size=256)
result = model.evaluate(x_test_pad, y_test)

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

text1 = "bu ürün çok iyi herkese tavsiye ederim"
text2 = "kargo çok hızlı aynı gün elime geçti"
text3 = "büyük bir hayal kırıklığı yaşadım bu ürün bu markaya yakışmamış"
text4 = "mükemmel"
text5 = "tasarımı harika ancak kargo çok geç geldi ve ürün açılmıştı tavsiye etmem"
text6 = "hiç resimde gösterildiği gibi değil"
text7 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
text8 = "hiç bu kadar kötü bir satıcıya denk gelmemiştim ürünü geri iade ediyorum"
text9 = "tam bir fiyat performans ürünü"
text10 = "beklediğim gibi çıkmadı"
texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]

tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
tokens_pad.shape

model.predict(tokens_pad)
