import pickle as pkl
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("model_pkl.pkl", "rb") as f:
    model = pkl.load(f)

with open("./sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []

for item in datastore:
    sentences.append(item['headline'])

training_sentences = sentences[0:training_size]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


def predict(model , sentences: list) -> None:
    sentences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return model.predict(padded)

st.title("Sentiment Analysis 🎭 for news headers")
st.info("Enter a news header to analyse it's mood (sarcastic/not sarcastic)")
sentence = st.text_input("Your sentence")  
result = predict(model, [sentence])

result = str(100 - float(str(result).split("[")[2].split("]")[0][:3])*100) + "%" + " Sarcastic"
st.text(result)


print("good")