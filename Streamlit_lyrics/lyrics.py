import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

model = tf.keras.models.load_model('lyrics_gen_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

st.title("Lyrics generator using NLP")

seed_text = st.text_input("Enter the first words for your song")
next_words = st.number_input("Enter the number of words for your song", min_value=0, format="%d")
st.write("Hit generate and get ready with your tune!")

if st.button("Generate"):

   for _ in range(next_words):
       token_list = tokenizer.texts_to_sequences([seed_text])[0]
       token_list = pad_sequences([token_list], maxlen = 19, padding = 'pre')
       predicted_probs = model.predict(token_list)[0]
       predicted = np.random.choice([x for x in range(len(predicted_probs))], p = predicted_probs)

       output_word = ' '
       for word, index in tokenizer.word_index.items():
           if index == predicted:
               output_word = word
               break
       seed_text += ' ' + output_word

   st.write(f"The generated lyrics is given below and it has {next_words} words (not as grammatical as modern day lyricists but still!!😁). Have fun!")
st.write(seed_text)