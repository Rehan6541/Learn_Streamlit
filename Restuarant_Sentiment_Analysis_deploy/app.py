import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter

# Download NLTK data
nltk.download('stopwords')

# Load model, vectorizer, and label encoder
model = joblib.load("Restaurant_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocess function
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return " ".join([ps.stem(word) for word in text if word not in stop_words])

# Streamlit UI
st.set_page_config(page_title="Restaurant Review Classifier", page_icon="🍽️")
st.title("🍽️ Restaurant Review Sentiment Classifier")
st.markdown("Enter a review and know whether it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your restaurant review here:", height=150)

if st.button("Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        # Preprocess the input review
        processed = preprocess(user_input)
        
        # Show review statistics
        word_count = len(processed.split())
        avg_word_length = np.mean([len(word) for word in processed.split()])
        st.write(f"**Review Length**: {word_count} words")
        st.write(f"**Average Word Length**: {avg_word_length:.2f} characters")

        # Word cloud for most frequent words
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed)
        st.image(wordcloud.to_array(), caption="Word Cloud of Your Review", use_column_width=True)
        
        # Vectorize the input and predict sentiment
        vector = vectorizer.transform([processed]).toarray()
        result = model.predict(vector)[0]
        result_label = label_encoder.inverse_transform([result])[0]
        
        # Show sentiment probability bar chart
        sentiment_probs = model.predict_proba(vector)[0]
        sentiment_labels = label_encoder.classes_

        # Plotting bar chart for sentiment probabilities
        fig, ax = plt.subplots()
        ax.bar(sentiment_labels, sentiment_probs, color=["#FF6666", "#66FF66"])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Probability")
        ax.set_title("Sentiment Prediction Probabilities")
        st.pyplot(fig)

        # Display result
        if result_label == "Positive":
            st.success(f"✅ Positive Review")
        else:
            st.error(f"❌ Negative Review")
        
        # Show more detailed word frequency information
        word_list = processed.split()
        word_freq = Counter(word_list)
        most_common_words = word_freq.most_common(10)
        
        # Display top 10 most frequent words
        st.write("### Top 10 Most Frequent Words")
        st.write(most_common_words)
        
        # Bar chart for word frequency
        words, counts = zip(*most_common_words)
        fig, ax = plt.subplots()
        ax.bar(words, counts, color="skyblue")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.set_title("Top 10 Words in the Review")
        st.pyplot(fig)
