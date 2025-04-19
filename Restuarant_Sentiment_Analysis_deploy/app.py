import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Download stopwords
nltk.download('stopwords')

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
    return data

data = load_data()

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return " ".join([ps.stem(word) for word in text if word not in stop_words])

# Process all reviews
corpus = [preprocess(review) for review in data["Review"]]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Liked"])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Restaurant Review Classifier", page_icon="🍽️")
st.title("🍽️ Restaurant Review Sentiment Classifier")
st.markdown("Enter a review and know whether it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your restaurant review here:", height=150)

if st.button("Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        # Preprocess input
        processed = preprocess(user_input)
        
        word_count = len(processed.split())
        avg_word_length = np.mean([len(word) for word in processed.split()])
        st.write(f"**Review Length**: {word_count} words")
        st.write(f"**Average Word Length**: {avg_word_length:.2f} characters")

        # Word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed)
        st.image(wordcloud.to_array(), caption="Word Cloud of Your Review", use_column_width=True)

        # Vectorize input
        input_vector = vectorizer.transform([processed]).toarray()
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector)[0]
        labels = label_encoder.classes_

        # Bar chart of probabilities
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color=["#FF6666", "#66FF66"])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Probability")
        ax.set_title("Sentiment Prediction Probabilities")
        st.pyplot(fig)

        # Show result
        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        # Word frequency
        word_freq = Counter(processed.split())
        top_words = word_freq.most_common(10)
        st.write("### Top 10 Most Frequent Words")
        st.write(top_words)

        words, counts = zip(*top_words)
        fig2, ax2 = plt.subplots()
        ax2.bar(words, counts, color="skyblue")
        ax2.set_xlabel("Words")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Top 10 Words in the Review")
        st.pyplot(fig2)
