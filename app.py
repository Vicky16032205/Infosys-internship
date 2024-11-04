%%writefile app.py
import streamlit as st
from PIL import Image
import base64
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tweepy
import requests
import re
import os
from datetime import datetime, timedelta

# Set up Twitter API
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAO9puQEAAAAAibBDDQEa%2BOYP%2BI%2BkQNdGzbfgOQQ%3DSryMUwRRPymuUcj0pA44K7TkHdRKXPE8LQrBeXkrQsIKgUHWDd"  # Use environment variable for security
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to fetch tweet text from URL
def fetch_tweet_text(url):
    tweet_id = url.split('/')[-1]
    tweet = client.get_tweet(tweet_id, tweet_fields=['text'])
    return tweet.data['text']

# Function to fetch recent tweets based on query
def fetch_recent_tweets(query, start_time, max_results=10):
    tweets = client.search_recent_tweets(query=query, start_time=start_time, tweet_fields=['text'], max_results=max_results)
    return [tweet.text for tweet in tweets.data]

# Load the dataset
df = pd.read_csv('/content/train.csv')  # Adjust path as needed

# Preprocess the text data
df['text'] = df['text'].str.lower()  # Convert text to lowercase
df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation and special characters
df['text'] = df['text'].str.replace('\d+', '', regex=True)  # Remove digits

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a RandomForest classifier for disaster detection
rf_disaster = RandomForestClassifier(n_estimators=100, random_state=42)
rf_disaster.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_disaster = rf_disaster.predict(X_test_tfidf)

# Evaluate the classifier
accuracy_disaster = accuracy_score(y_test, y_pred_disaster)
report_disaster = classification_report(y_test, y_pred_disaster)

# Load the background image from URL
def get_image_as_base64(image_url):
    response = requests.get(image_url)
    img_base64 = base64.b64encode(response.content).decode()
    return img_base64

# URL of the background image
image_url = "https://static.vecteezy.com/system/resources/previews/022/874/470/non_2x/wildfire-forest-burning-4k-digital-painting-illustration-of-trees-that-burn-wild-flames-raging-trough-the-environment-background-wallpaper-red-flames-generate-ai-free-photo.jpg"

# Convert image to base64
img_base64 = get_image_as_base64(image_url)

# Set background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url(data:image/jpeg;base64,{img_base64});
    background-size: cover;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title
st.title("Tweet Classifier")

# Text input for tweet or Twitter query
query = st.text_input("Enter your tweet or Twitter URL here:")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Function to classify text
def classify_text(text):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = rf_disaster.predict(text_tfidf)
    return 'DISASTER' if prediction[0] == 1 else 'NOT A DISASTER'

# Button to classify tweet or URL
if st.button("Classify"):
    if query.startswith('http'):
        # Handle Twitter URL
        try:
            tweet_text = fetch_tweet_text(query)
            st.write(f"Original Tweet: {tweet_text}")
            classification = classify_text(tweet_text)
            st.write(f"Predicted as Disaster-related: {classification}")
        except Exception as e:
            st.write(f"Error: {e}")
    else:
        # Handle plain text
        classification = classify_text(query)
        st.write(f"Classifying text: {query}")
        st.write(f"Predicted as Disaster-related: {classification}")

# Time range selection for fetching recent tweets
time_range = st.selectbox("Select the time range for tweets:", ("5 hours", "24 hours"))

# Button to fetch and classify recent disaster-related tweets
if st.button("Search Recent Disaster Tweets"):
    try:
        if time_range == "5 hours":
            start_time = datetime.utcnow() - timedelta(hours=5)
        elif time_range == "24 hours":
            start_time = datetime.utcnow() - timedelta(hours=24)

        start_time = start_time.isoformat("T") + "Z"
        recent_tweets = fetch_recent_tweets("disaster", start_time, max_results=10)

        st.write(f"Recent tweets in the last {time_range}:")
        for tweet in recent_tweets:
            classification = classify_text(tweet)
            with st.expander(f"Tweet: {tweet[:30]}..."):
                st.write(f"Tweet: {tweet}")
                st.write(f"Predicted as Disaster-related: {classification}")
    except Exception as e:
        st.write(f"Error: {e}")
