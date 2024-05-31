from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data


def train_model(file_path: str):
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(file_path)

    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test_tfidf)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    file_path = 'data/conversations.csv'
    model, vectorizer = train_model(file_path)
