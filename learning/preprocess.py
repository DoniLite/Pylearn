import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def preprocess_data_for_conversation():
    data = pd.read_csv('dataset/Conversation.csv')

    X = data['question']
    y = data['answer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def preprocess_data_for_location():
    data = pd.read_csv('dataset/location.csv')

    X = data['question']
    y = data['answer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
