import pandas as pd
from sklearn.metrics import confusion_matrix


def filter_data(model, vectorizer, X_train, y_train):
    y_pred_train = model.predict(X_train)
    cm = confusion_matrix(y_train, y_pred_train)

    # Supposons que nous voulons filtrer les exemples mal classifiés
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_train, y_pred_train)) if true != pred]

    filtered_data = pd.DataFrame({
        'message': X_train[misclassified_indices],
        'response': y_train[misclassified_indices]
    })

    return filtered_data


def main():
    file_path = 'data/conversations.csv'
    model, vectorizer, X_train, X_test, y_train, y_test = train_model(file_path)

    filtered_data = filter_data(model, vectorizer, X_train, y_train)
    filtered_data.to_csv('data/filtered_conversations.csv', index=False)

    print(f"Filtered data saved to 'data/filtered_conversations.csv'")


if __name__ == "__main__":
    main()
