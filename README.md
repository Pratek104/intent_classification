# Intent Classification with SVM and SpaCy

This project demonstrates how to perform intent classification using a Support Vector Machine (SVM) classifier and the SpaCy library for text embeddings. The model predicts the intent of a user’s input sentence, such as checking the weather or making a call, based on a pre-trained model.

## Project Overview

- **Text Data**: The dataset contains user inputs (sentences) and their corresponding intents.
- **Model**: A Support Vector Machine (SVM) classifier is used to classify the text into one of the predefined intents.
- **Text Representation**: Sentences are converted into vector representations using the `en_core_web_lg` SpaCy model.
- **Deployment**: The model is deployed via a command-line interface (CLI) for easy testing of predictions.

## Features

- **Sentence Encoding**: Converts input text into vectors using SpaCy's pre-trained large English model (`en_core_web_lg`).
- **Intent Classification**: Uses a Support Vector Machine (SVM) for training and prediction of text intents.
- **Prediction Demo**: A simple command-line interface (CLI) allows users to input sentences and see predicted intents.
- **Model Persistence**: The trained SVM model and label encoder are saved as `.pkl` files for future use.

## Requirements

- Python 3.x
- SpaCy (`en_core_web_lg` model)
- scikit-learn
- pandas
- numpy

### Install Dependencies

Before running the project, install the required dependencies:

```bash
pip install spacy scikit-learn pandas numpy
python -m spacy download en_core_web_lg
