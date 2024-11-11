import numpy as np
import pandas as pd
import spacy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import csv

# Load spaCy model for sentence encoding
nlp = spacy.load("en_core_web_lg")

# Function to read the CSV data
def read_data(path):
    labels = []
    sentences = []
    with open(path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)  # Skip header row
        for row in readCSV:
            labels.append(row[0])
            sentences.append(row[1])
    return sentences, labels

# Load data (change to your actual data file path)
data_file_path = 'data.csv'
sentences, labels = read_data(data_file_path)



# Split data into train and test sets
sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Encode sentences into vectors using spaCy
def encode_sentences(sentences):
    n_sentences = len(sentences)
    embedding_dim = nlp.vocab.vectors_length
    X = np.zeros((n_sentences, embedding_dim))
    for idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        X[idx, :] = doc.vector
    return X

train_X = encode_sentences(sentences_train)
test_X = encode_sentences(sentences_test)

# Label Encoding
def label_encoding(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le

train_y, le = label_encoding(labels_train)
test_y, _ = label_encoding(labels_test)

# Train the SVM model
def svc_training(X, y):
    clf = SVC(C=1)
    clf.fit(X, y)
    return clf

model = svc_training(train_X, train_y)

# Evaluate model performance on test set
def evaluate_model(model, X, y):
    y_pred = model.predict(X)


evaluate_model(model, train_X, train_y)
evaluate_model(model, test_X, test_y)

# Save the model and label encoder to pickle
with open('svc_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

print("Model and label encoder have been saved.")

# Function to predict intent using the trained model
def predict_intent(text, model, label_encoder):
    doc = nlp(text)
    text_vector = doc.vector.reshape(1, -1)  
    label_encoded = model.predict(text_vector)  
    label_decoded = label_encoder.inverse_transform(label_encoded) 
    return label_decoded[0]

# Test the prediction function
text = "read todays news"
predicted_intent = predict_intent(text, model, le)
print(f"Predicted intent for '{text}': {predicted_intent}")

# Command-line interface for demo
if __name__ == "__main__":
    print("Enter a sentence, and the model will predict the intent.\n")
    while True:
        user_input = input("Enter a sentence (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predicted_intent = predict_intent(user_input, model, le)
        print(f"Predicted Intent: {predicted_intent}\n")
