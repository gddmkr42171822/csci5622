"""
TODO: Add stopwords from nltk or sklearn.
TODO: Add stemming from nltk (SnowballStemmer).
TODO: Try TfIdfVectorizer from sklearn.
"""
import csv
import sys
import random
import re
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
# from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score  ######################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn import metrics

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Prevent the csv reader from throwing an error about the size of the
# field it is reading in.
csv.field_size_limit(sys.maxsize)


def GetSamples(files):
    X = []
    y = []
    for f in files:
        with open(f, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                X.append(row[1])
                y.append(row[0])
    return (np.array(X), np.array(y))


def RemoveStopWords(stop_words, word_list):
    filtered_text = [
        word for word in word_list if word.lower() not in stop_words]
    return filtered_text


def StemText(stemmer, word_list):
    stemmed_text = [stemmer.stem(word) for word in word_list]
    # Remove duplicate strings in the stemmed_text list while keeping order.
    return list(OrderedDict.fromkeys(stemmed_text))


def KeepWordsSpaces(text):
    regex = re.compile(r'[^a-zA-Z ]')
    text = regex.sub('', text)
    return text


def PreprocessSamples(X):
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    X_modified = []
    # Preprocess the training examples: remove stop words, stem the words, etc.
    for line in X:
        line = KeepWordsSpaces(line)
        # Tokenize the line into a list of words.
        line = line.split()
        line = RemoveStopWords(stop_words, line)
        line = StemText(stemmer, line)
        # Rejoin the list of tokens in to a single string.
        line = ' '.join(line)
        X_modified.append(line)
    return np.array(X_modified)


def MultinomialNaiveBayes(X, y):
    clf = MultinomialNB()
    print 'Multinomial Naive Bayes score', cross_val_score(clf, X, y).mean()
    clf.fit(X, y)
    return clf


def DecisionTreeStump(X, y):
    clf = DecisionTreeClassifier(max_depth=1)
    print 'Decision Tree Stump score', cross_val_score(clf, X, y).mean()
    clf.fit(X, y)
    return clf


def AdaBoostDecisionTree(X, y):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    print 'AdaBoost Decision Tree score', cross_val_score(clf, X, y).mean()
    clf.fit(X, y)
    return clf


def PerceptronPrediction(X, y):
    clf = Perceptron()
    print 'Perceptron Prediction score', cross_val_score(clf, X, y).mean()
    clf.fit(X, y)
    return clf


def AdaboostPerceptron(X, y):
    clf = AdaBoostClassifier(Perceptron(), algorithm='SAMME')
    print 'Adaboost Perceptron score', cross_val_score(clf, X, y).mean()
    clf.fit(X, y)
    return clf


from collections import Counter


def majority_vote_classifier(classifiers_list, X):
    tmp = []
    for clf in classifiers_list:
        tmp.append(clf.predict(X))
    y = []
    for ii in xrange(X.shape[0]):
        lst = [t[ii] for t in tmp]
        c = Counter(lst)
        y.append(c.most_common(1)[0][0])
    return y


def balance_samples(X, y, max = 1000):
    categories = []
    for line in y:
        if not line in categories:
            categories.append(line)

    new_x = []
    new_y = []

    counter_cat = np.zeros(len(categories))

    for xx, yy in zip(X, y):
        for ii in range(len(categories)):
            if yy == categories[ii]:
                if counter_cat[ii] >= max:
                    break
                else:
                    counter_cat[ii] += 1
                    new_x.append(xx)
                    new_y.append(yy)

    return new_x ,  new_y


from sklearn.base import BaseEstimator, TransformerMixin
import nltk


class PoS_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_feature_names(self):
        PoS_tags = ['PRP$', 'VBG', 'VBD', '``', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', '$',
                    'NN', ')', '(', 'FW', 'POS', '.', 'TO', 'LS', 'RB', ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT',
                    'RBS', 'RBR', 'CD', 'PRP', 'EX', 'IN', 'WP$', 'MD', 'NNPS', '--', 'JJS', 'JJR', 'SYM', 'UH']
        return PoS_tags

    def fit(self, examples):
        # return self and nothing else
        return self

    def transform(self, examples):
        import numpy as np
        from scipy.sparse import csr_matrix

        new_examples = []

        for line in examples:
            line = line.split()
            line_pos_tag = nltk.pos_tag(line)
            line_pos_tag = [tag for (word, tag) in line_pos_tag]
            line = ' '.join(line_pos_tag)
            new_examples.append(line)

        PoS_tags = ['PRP$', 'VBG', 'VBD', '``', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', '$',
                    'NN', ')',
                    '(', 'FW', 'POS', '.', 'TO', 'LS', 'RB', ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR',
                    'CD',
                    'PRP', 'EX', 'IN', 'WP$', 'MD', 'NNPS', '--', 'JJS', 'JJR', 'SYM', 'UH']

        # Initiaize matrix
        X = np.zeros((len(new_examples), len(PoS_tags)))

        # Loop over examples and count letters
        for ii, x in enumerate(new_examples):
            X[ii, :] = np.array([x.count(tag) for tag in PoS_tags])

        return csr_matrix(X)


def main():
    # Create a list of the relatives paths to all of the csv files.
    samples = []
    for f in listdir('csv_data/'):
        f = join('csv_data/', f)
        if isfile(f):
            samples.append(f)

    X, y = GetSamples(samples)
    # X = PreprocessSamples(X)

    X, y = balance_samples(X, y)


    X, y = shuffle(X, y, random_state=random.randint(1, 100))

    from sklearn.pipeline import FeatureUnion
    allmyfeatures = FeatureUnion([
        ("bag-of-words", TfidfVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
        ("PoS-counts", PoS_Transformer())
    ])

    X = allmyfeatures.fit_transform(X)

    classifiers_list = []

    MNB_clf = MultinomialNaiveBayes(X, y)
    classifiers_list.append(MNB_clf)
    #print("MultinomialNaiveBayes Accuracy: %0.4f" % metrics.accuracy_score(y, MNB_clf.predict(X)))

    DTS_clf = DecisionTreeStump(X, y)
    classifiers_list.append(DTS_clf)
    #print("DecisionTreeStump Accuracy: %0.4f" % metrics.accuracy_score(y, DTS_clf.predict(X)))

    ABDT_clf = AdaBoostDecisionTree(X, y)
    classifiers_list.append(ABDT_clf)
    #print("AdaBoostDecisionTree Accuracy: %0.4f" % metrics.accuracy_score(y, ABDT_clf.predict(X)))

    PP_clf = PerceptronPrediction(X, y)
    classifiers_list.append(PP_clf)
    #print("PerceptronPrediction Accuracy: %0.4f" % metrics.accuracy_score(y, PP_clf.predict(X)))

    AP_clf = AdaboostPerceptron(X, y)
    classifiers_list.append(AP_clf)
    #print("AdaboostPerceptron Accuracy: %0.4f" % metrics.accuracy_score(y, AP_clf.predict(X)))

    predicted = majority_vote_classifier(classifiers_list, X)
    print("majority_vote_classifier Accuracy: %0.4f" % metrics.accuracy_score(y, predicted))


if __name__ == '__main__':
    main()
