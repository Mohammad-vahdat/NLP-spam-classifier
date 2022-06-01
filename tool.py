### Import packages:
import pandas as pd
import string
import nltk

# import matplotlib.pyplot as plt
# import seaborn as sns


class spamClassifier:
    def __init__(self, messages):

        self.messages = messages

    def generate_tfidf(self):

        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        bow_transformer = CountVectorizer(analyzer=self.text_process).fit(
            self.messages["message"]
        )

        print("number of unique words:", len(bow_transformer.vocabulary_))

        messages_bow = bow_transformer.transform(self.messages["message"])

        print("shape of sparse matrix:", messages_bow.shape)
        print("amount of non-zero occurances:", messages_bow.nnz)

        sparcity = 100 * (
            messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])
        )

        print("sparcity:", sparcity)

        tfidf_transfromer = TfidfTransformer().fit(messages_bow)

        messages_tdidf = tfidf_transfromer.transform(messages_bow)

        print("shape of message tfidf:", messages_tdidf.shape)

        return messages_tdidf

    def text_process(self, message):

        from nltk.corpus import stopwords

        nonpunc = [char for char in message if char not in string.punctuation]

        nonpunc = "".join(nonpunc)

        return [
            word
            for word in nonpunc.split()
            if word.lower() not in stopwords.words("english")
        ]

    def naive_bayes_builder(self, msg_train, label_train):

        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB().fit(msg_train, label_train)

    def LSTM_(self, msg_train, label_train):

        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Dense, LSTM, Embedding, Activation, Dropout

        # from tensorflow.keras.optimizers import adam

        model = Sequential()
        model.add(Embedding(5572, 11425, input_length=11425))
        model.add(LSTM(100))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="RMSprop", metrics=["accuracy"], loss="binary_crossentropy"
        )

        print(model.summary())

        return model.fit(msg_train, label_train, epochs=25)

    def evaluation(self, label_test, predictions):

        from sklearn.metrics import classification_report

        print("classification report:", classification_report(label_test, predictions))

        return
