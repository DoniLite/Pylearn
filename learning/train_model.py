from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data_for_conversation
import joblib


def train_model():
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data_for_conversation()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test
