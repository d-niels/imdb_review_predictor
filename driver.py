import os
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model


# create a text file and writes to it
def create_txt(name, output):
    text_file = open(name, "w")
    for x in output:
        text_file.write(str(x.astype(int)[0]) + "\n")
    text_file.close()


# keeps track of data and runs different machine learning algorithms on it
# and then tests their effectiveness at predicting
# outputs details to a text file
class Language_Processor():
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.vectorizer = None
        self.classifier = None

        self.uni = False
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = False

    # predicts using whichever algorithm has been set and saves results to a text file
    def predict(self):
        output = []
        for i in range(len(self.x_test)):
            output.append(self.classifier.predict(self.vectorizer.transform([self.x_test[i]])))

        if self.uni:
            create_txt("unigram.output.txt", output)
        if self.bi:
            create_txt("bigram.output.txt", output)
        if self.uni_tfidf:
            create_txt("unigramtfidf.output.txt", output)
        if self.bi_tfidf:
            create_txt("bigramtfidf.output.txt", output)

    def train_uni_SGD(self):
        print("TRAINING UNI SGD")
        self.uni = True
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = False

        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_bi_SGD(self):
        print("TRAINING BI SGD")
        self.uni = False
        self.bi = True
        self.uni_tfidf = False
        self.bi_tfidf = False

        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_uni_tfidf_SGD(self):
        print("TRAINING TFIDF UNI SGD")
        self.uni = False
        self.bi = False
        self.uni_tfidf = True
        self.bi_tfidf = False

        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)

    def train_bi_tfidf_SGD(self):
        print("TRAINING TFIDF BI SGD")
        self.uni = False
        self.bi = False
        self.uni_tfidf = False
        self.bi_tfidf = True

        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        self.classifier = linear_model.SGDClassifier(loss="hinge", penalty="l1")
        self.classifier.fit(self.vectorizer.fit_transform(self.x_train), self.y_train)


if __name__ == "__main__":

    # get the training data
    train = pd.read_csv("imdb_tr.csv")
    x_train = train["text"]
    y_train = train["polarity"]

    # get the testing data
    test = pd.read_csv("imdb_te.csv")
    x_test = test["text"]

    # create language processor
    lp = Language_Processor(x_train, y_train, x_test)

    # train and test on different classifiers
    lp.train_uni_SGD()
    lp.predict()

    lp.train_bi_SGD()
    lp.predict()

    lp.train_uni_tfidf_SGD()
    lp.predict()

    lp.train_bi_tfidf_SGD()
    lp.predict()
