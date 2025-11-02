import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

st.title("Twitter Sentiment Analysis Demo")
st.write("Enter a tweet to predict its sentiment:")

# Download NLTK stopwords if not present
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    porter = PorterStemmer()
    text = text.lower()
    text = ''.join([i for i in text if i in string.ascii_lowercase+' '])
    text = ' '.join([porter.stem(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def predict_sentiment(text):
    # Replace this with your notebook's actual model logic!
    import random
    return random.choice(["positive", "negative", "neutral"])

tweet = st.text_area("Tweet text", "")

if st.button("Analyze Sentiment") and tweet.strip():
    preprocessed = preprocess_text(tweet)
    sentiment = predict_sentiment(preprocessed)
    st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")

st.info("Note: Replace the dummy prediction with your actual trained model and vectorizer logic for real predictions.")