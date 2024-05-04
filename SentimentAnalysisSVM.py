from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd


file_path = "data/movie.csv"
data = pd.read_csv(file_path)


trainData, testData = train_test_split(data, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(trainData['text'])
test_vectors = vectorizer.transform(testData['text'])

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, trainData['label'])

input_file_path = "test_file.txt" 
with open(input_file_path, 'r') as file:
    input_text = file.read().splitlines()

input_data = {'text': input_text}
input_df = pd.DataFrame(input_data)
input_vectors = vectorizer.transform(input_df['text'])

prediction_linear = classifier_linear.predict(input_vectors)

if prediction_linear[0] == 1:
    print("Sentiment: Positive")
elif prediction_linear[0] == 0:
    print("Sentiment: Negative")



test_predictions = classifier_linear.predict(test_vectors)
print("\nClassification Report for Test Data:")
print(classification_report(testData['label'], test_predictions))
