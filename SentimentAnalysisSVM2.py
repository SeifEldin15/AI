import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Assuming you have a CSV file with columns 'review' and 'sentiment'
df = pd.read_csv('2.csv')
X = df['review']
y = df['sentiment']
X_processed = X.apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X_processed)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_vectorized, y)

# Take user input
user_input = input("Enter your text to test sentiment: ")
processed_input = preprocess_text(user_input)
vectorized_input = vectorizer.transform([processed_input])

# Predict sentiment
predicted_sentiment = svm_classifier.predict(vectorized_input)[0]
if predicted_sentiment == 1:
    print("Positive sentiment detected.")
else:
    print("Negative sentiment detected.")
