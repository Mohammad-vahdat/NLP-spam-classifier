### Import packages:
import pandas as pd
import string
import nltk

# import matplotlib.pyplot as plt
# import seaborn as sns


### Read messages:
messages = pd.read_csv(
    "smsspamcollection/SMSSpamCollection", sep="\t", names=["label", "message"]
)

messages["length"] = messages["message"].apply(len)

# print(messages.head())
# print(messages.groupby("label").describe())

# plt.figure()
# sns.distplot(messages["length"], kde=False)
# plt.show()


def text_process(message):

    from nltk.corpus import stopwords

    nonpunc = [char for char in message if char not in string.punctuation]

    nonpunc = "".join(nonpunc)

    return [
        word
        for word in nonpunc.split()
        if word.lower() not in stopwords.words("english")
    ]


# print(messages["message"].head().apply(text_process))

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])

print("number of unique words:", len(bow_transformer.vocabulary_))

messages_bow = bow_transformer.transform(messages["message"])

print("shape of sparse matrix:", messages_bow.shape)
print("amount of non-zero occurances:", messages_bow.nnz)

sparcity = 100 * (messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print("sparcity:", sparcity)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transfromer = TfidfTransformer().fit(messages_bow)

messages_tdidf = tfidf_transfromer.transform(messages_bow)

print("shape of message tfidf:", messages_tdidf.shape)

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(
    messages_tdidf, messages["label"], test_size=0.2
)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(msg_train, label_train)

predictions = spam_detect_model.predict(msg_test)

from sklearn.metrics import classification_report

print("classification report:", classification_report(label_test, predictions))
