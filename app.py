import pickle as pkl
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re

max_length = 100
trunc_type='post'
padding_type='post'

with open("model_pkl.pkl", "rb") as f:
    model = pkl.load(f)


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)

def predict(model , sentences: list) -> None:
    sentences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return model.predict(padded)

st.title("Sentiment Analysis 🎭 for news headers")
st.info("Enter a news header to analyse it's mood (sarcastic/not sarcastic)")
sentence = st.text_input("Your sentence")
res_pattern = re.compile(r"\d\.\d\d")
result = predict(model, [sentence])


st.text(f"{float(re.findall(res_pattern, str(result))[0])*100} % sarcastic")