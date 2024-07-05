import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

with open('./bow.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('./NB_bow.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./ps.pkl', 'rb') as f:
    ps = pickle.load(f)

def preprocessText(text):
    text = re.sub('[^A-Za-z0-9]', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(stemmed_text)

st.title("Spam Message Detection")

user_msg = st.text_input("Enter the message")
user_msg = preprocessText(user_msg)

user_submit = st.button("Predict")

if user_submit:
    user_msg = cv.transform([user_msg])
    prediction = model.predict(user_msg)

    if prediction == 0:
        prediction_text = "Not Spam"
    else:
        prediction_text = "Spam"

    st.subheader("The message is : ")
    st.write(prediction_text)