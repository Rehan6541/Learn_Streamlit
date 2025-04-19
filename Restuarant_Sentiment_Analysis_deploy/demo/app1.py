import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem.porter import PorterStemmer

# Set Streamlit page config
st.set_page_config(page_title="Restaurant Review Classifier", page_icon="🍽️")

# Basic English stopwords (manual, to avoid NLTK download errors)
stop_words = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now"
}

# Stemmer
ps = PorterStemmer()

# Preprocess text
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return " ".join([ps.stem(word) for word in text if word not in stop_words])

# Load data and train model
data = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
data.dropna(inplace=True)
data['cleaned'] = data['Review'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(data['cleaned']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Liked'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI
st.title("🍽️ Restaurant Review Sentiment Classifier")
st.markdown("Enter a restaurant review to predict whether it is **Positive** or **Negative**.")

user_input = st.text_area("Enter your restaurant review here:", height=150)

if st.button("Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(user_input)

        # Stats
        word_count = len(processed.split())
        avg_word_len = np.mean([len(word) for word in processed.split()])
        st.write(f"**Review Length**: {word_count} words")
        st.write(f"**Average Word Length**: {avg_word_len:.2f} characters")

        # Wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(processed)
        st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

        # Predict
        vector = vectorizer.transform([processed]).toarray()
        result = model.predict(vector)[0]
        result_label = label_encoder.inverse_transform([result])[0]

        # Sentiment Result
        if result_label == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        # Top Words
        word_freq = Counter(processed.split())
        most_common = word_freq.most_common(10)
        words, counts = zip(*most_common)

        fig, ax = plt.subplots()
        ax.bar(words, counts, color="skyblue")
        ax.set_title("Top 10 Words in Your Review")
        st.pyplot(fig)
