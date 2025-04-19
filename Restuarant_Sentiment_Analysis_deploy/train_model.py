import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')  # Ensure dataset has "Liked" column (1=Positive, 0=Negative)

# Text preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return " ".join([ps.stem(word) for word in text if word not in stop_words])

corpus = [preprocess(review) for review in data["Review"]]

# Vectorization (TF-IDF for better feature extraction)
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(corpus).toarray()

# Encoding the labels (1=Positive, 0=Negative)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Liked"])  # 0=Negative, 1=Positive

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Voting Classifier to combine the best models
voting_clf = VotingClassifier(estimators=[
    ('logreg', models["Logistic Regression"]),
    ('nb', models["Naive Bayes"]),
    ('rf', models["Random Forest"]),
    ('svm', models["SVM"]),
    ('knn', models["KNN"])
], voting='soft')

models['Voting Classifier'] = voting_clf

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Choose best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} (Accuracy: {scores[best_model_name]:.4f})")

# Save the best model, vectorizer, and label encoder
joblib.dump(best_model, "Restaurant_review_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Best model, vectorizer, and label encoder saved successfully.")
