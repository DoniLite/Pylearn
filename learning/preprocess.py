import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(file_path: str):
    data = pd.read_csv(file_path)

    # Séparation des données en features et target
    X = data['message']
    y = data['label']

    # Division des données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorisation des messages
    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test, vectorizer
