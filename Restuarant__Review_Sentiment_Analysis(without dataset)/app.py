import streamlit as st
from textblob import TextBlob
import re
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Restaurant Review Sentiment", layout="centered")
st.title("🍽️ Restaurant Review Sentiment Analyzer")

st.write("Share your thoughts about a restaurant, and we'll tell you how positive, negative, or neutral the review sounds!")

user_input = st.text_area("Enter your restaurant review", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a restaurant review first.")
    else:
        cleaned = clean_text(user_input)
        polarity = TextBlob(cleaned).sentiment.polarity
        
        # Display sentiment
        if polarity > 0:
            sentiment = "😊 Positive"
            st.success(f"Sentiment: {sentiment}")
        elif polarity < 0:
            sentiment = "😠 Negative"
            st.error(f"Sentiment: {sentiment}")
        else:
            sentiment = "😐 Neutral"
            st.info(f"Sentiment: {sentiment}")
        
        # Show polarity score
        st.write(f"Polarity Score: `{polarity}`")
