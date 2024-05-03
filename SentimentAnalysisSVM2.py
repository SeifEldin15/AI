from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
import pandas as pd

# Load the text file
file_path = input("Enter the path to the text file: ")
with open(file_path, 'r') as file:
    text = file.read()

# Create a DataFrame with the text
data = {'Content': [text]}
input_data = pd.DataFrame(data)

# Load the training data and test data
trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])
input_vectors = vectorizer.transform(input_data['Content'])

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, trainData['Label'])
prediction_linear = classifier_linear.predict(input_vectors)

if prediction_linear[0] == 'pos':
    print("Sentiment: Positive")
elif prediction_linear[0] == 'neg':
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")

# Generate classification report for test data
test_predictions = classifier_linear.predict(test_vectors)
print("\nClassification Report for Test Data:")
print(classification_report(testData['Label'], test_predictions))
