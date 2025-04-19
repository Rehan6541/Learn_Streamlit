import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Set Streamlit page config
st.set_page_config(page_title="Restaurant Review Classifier", page_icon="🍽️")

# Setup NLTK and preprocessing
nltk.download("stopwords")
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return " ".join([ps.stem(word) for word in text if word not in stop_words])

# Load and preprocess dataset
@st.cache_data
def load_and_train():
    data = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
    data['cleaned'] = data['Review'].apply(preprocess)

    tfidf = TfidfVectorizer(max_features=1500)
    X = tfidf.fit_transform(data['cleaned']).toarray()

    le = LabelEncoder()
    y = le.fit_transform(data['Liked'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, tfidf, le

# Train the model during app startup
model, vectorizer, label_encoder = load_and_train()

# Streamlit UI
st.title("🍽️ Restaurant Review Sentiment Classifier")
st.markdown("Enter a review and see if it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your restaurant review here:", height=150)

if st.button("Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(user_input)

        # Show review stats
        word_count = len(processed.split())
        avg_word_len = np.mean([len(word) for word in processed.split()])
        st.write(f"**Review Length**: {word_count} words")
        st.write(f"**Average Word Length**: {avg_word_len:.2f} characters")

        # Wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(processed)
        st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

        # Vectorize and predict
        vector = vectorizer.transform([processed]).toarray()
        result = model.predict(vector)[0]
        result_label = label_encoder.inverse_transform([result])[0]

        # Show sentiment
        if result_label == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        # Word frequency bar chart
        word_freq = Counter(processed.split())
        most_common = word_freq.most_common(10)
        words, counts = zip(*most_common)

        fig, ax = plt.subplots()
        ax.bar(words, counts, color="skyblue")
        ax.set_title("Top 10 Words in Your Review")
        st.pyplot(fig)
