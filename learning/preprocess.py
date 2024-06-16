import pandas as pd
from torch.utils.data import DataLoader
from datasetClass import ConversationDataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


conv_training_data = ConversationDataset(data_file_path='dataset/Conversation.csv', tokenizer=tokenizer)
train_dataloader = DataLoader(conv_training_data, batch_size=64, shuffle=True)


def preprocess_data_for_conversation():
    # data = pd.read_csv('dataset/Conversation.csv')
    # print(data.size)
    # print(data)
    print(train_dataloader)
    for i, (x, y) in enumerate(train_dataloader):
        print(f'Batch {i + 1}')
        print(x)
        print(y)


preprocess_data_for_conversation()


# def preprocess_data_for_location_v1():
#     data = pd.read_csv('dataset/location.csv')
#
#     X = data['question']
#     y = data['answer']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     vectorizer = TfidfVectorizer()
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)
#
#     return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
#

# def preprocess_data_for_location_v2():
#     data = pd.read_csv('dataset/location.csv')
#
#     X = data['some']
#     y = data['some']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     vectorizer = TfidfVectorizer()
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)
#
#     return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


