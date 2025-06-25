import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model and tokenizer
model = load_model('lyrics_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_seq_len = 50  # Use the same value as in training

def generate_lyrics(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                output_word = word
                break
        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

st.title("Lyrics Generator")
prompt = st.text_input("Enter a prompt or a few lines:")
num_words = st.slider("Number of words to generate", 10, 100, 50)
if st.button("Generate Lyrics"):
    st.write(generate_lyrics(prompt, next_words=num_words))