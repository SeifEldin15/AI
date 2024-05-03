# Sentiment Analysis using Support Vector Machines (SVM)

This repository contains Python code for performing sentiment analysis using SVM with TF-IDF feature extraction.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/seifeldin15/sentiment-analysis-svm.git
   cd sentiment-analysis-svm
   
pip install pandas scikit-learn

python sentiment_analysis.py

######Code Explanation

Import Libraries: The code imports necessary libraries like TfidfVectorizer for feature extraction, classification_report for evaluation, and svm for SVM classification.
Load Text Data: Users can input the path to the text file, which is then loaded into a Pandas DataFrame.
Load Training and Test Data: Training and test data are loaded from CSV files (train.csv and test.csv).
Feature Extraction: TF-IDF vectors are created using TfidfVectorizer to represent text data.
SVM Classification: SVM with a linear kernel is used for sentiment classification.
Generate Classification Report: A classification report is generated for evaluating the model's performance on the test data.

######Files

sentiment_analysis.py: Main Python script containing the sentiment analysis code.
train.csv: CSV file containing training data.
test.csv: CSV file containing test data.
README.md: This README file explaining the project and usage instructions.

######License

This project is licensed under the MIT License - see the LICENSE file for details.
