"""
Baseline score: 0.63821
5% over baseline score: .67
10% over baseline score: .70

With bigrams also in the features:
Score: 0.64499

With the counts of parts of speech of only words in the sentence:
Score: 0.47967

With the counts of a combination of the part of speech and word in the sentence while stemming and removing punctuation:
Score: 0.61789

With 4-grams and stemming while removing punctuation and numbers:
Score: 0.66396

With Tfidf vectorizer:
Score: 0.64905

"""
import re
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


def KeepWordsAndSpaces(text):
    punctuation_numbers_specialcharacters = re.compile(r'[^a-zA-Z ]')
    text = punctuation_numbers_specialcharacters.sub('', text)
    return text.lower()


class StemmerTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()
    def __call__(self, text):
        return [self.ps.stem(t) for t in word_tokenize(text)]


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, text):
        return [self.wnl.lemmatize(t) for t in word_tokenize(text)]


class POSCountTokenizer(object):
    def __call__(self, text):
        words = word_tokenize(text)
        words_and_pos_tags = pos_tag(words)
        # Return a list of the part of speech for each word in the sentence.
        return [word_and_pos[1] for word_and_pos in words_and_pos_tags]


class POSWordAssociationTokenizer(object):
    def __call__(self, text):
        words = word_tokenize(text)
        words_and_pos_tags = pos_tag(words)
        return [word_and_pos[0] + '=' + word_and_pos[1] for word_and_pos in words_and_pos_tags]


def SentenceLength(text):
    text = KeepWordsAndSpaces(text)
    words = word_tokenize(text)
    yield len(words)


class Featurizer:
    def __init__(self): 
        # Vectorizing by 4-gram word count.
        self.vectorizer = CountVectorizer(
            tokenizer=LemmaTokenizer(),
            ngram_range=(1,8),
            preprocessor=KeepWordsAndSpaces)

        # Vectorizing with the count of the part of speech in the sentence.
        # self.vectorizer = CountVectorizer(
        #     preprocessor=KeepWordsAndSpaces,
        #     tokenizer=POSCountTokenizer(),
        #     ngram_range=(1,4))

        # Vectorizing with a combination of the word and pos in the sentence.
        # self.vectorizer = CountVectorizer(
        #     preprocessor=KeepWordsAndSpaces,
        #     tokenizer=POSWordAssociationTokenizer(),
        #     ngram_range=(1,8))

        # Using a TfidfVectorizer instead of a CountVectorizer.
        # self.vectorizer = TfidfVectorizer(
        #     tokenizer=StemmerTokenizer(),
        #     ngram_range=(1,4),
        #     preprocessor=KeepWordsAndSpaces)

        # Using sentence length as the feature
        # self.vectorizer = CountVectorizer(
        #     analyzer=SentenceLength)

    def train_feature(self, examples):
        X = self.vectorizer.fit_transform(examples)
        # print X.toarray()
        return X

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())

        for feature in feature_names:
            print feature

        # if len(categories) == 2:
        #     top10 = np.argsort(classifier.coef_[0])[-10:]
        #     bottom10 = np.argsort(classifier.coef_[0])[:10]
        #     print("Pos: %s" % " ".join(feature_names[top10]))
        #     print("Neg: %s" % " ".join(feature_names[bottom10]))
        # else:
        #     for i, category in enumerate(categories):
        #         top10 = np.argsort(classifier.coef_[i])[-10:]
        #         print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    validation = list(DictReader(open("../data/spoilers/validation.csv", 'r')))

    feat = Featurizer()
    labels = []
    
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    # print("Label set: %s" % str(labels))
    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    y_validation = array(list(labels.index(x[kTARGET_FIELD])
                         for x in validation))

    x_validation = feat.test_feature(x[kTEXT_FIELD] for x in validation)

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    print 'Accuracy', accuracy_score(y_validation, predictions)

    # if predictions[i] != y_validation[i]:
    #     print '%d labelled incorrectly.  Should be %r instead of %r' % (i, y_validation[i], predictions[i])

    # print 'Cross validation score: ', cross_val_score(
    #     lr, x_validation, y_validation, cv=5).mean()
    
    # print '\t' + '\t'.join(labels)
    # for i, row in enumerate(confusion_matrix(y_validation, predictions)):
    #     print labels[i] + '\t' + '\t'.join(str(column) for column in row)

    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)

    # Find average number of words in a spoiler and not a spoiler.
    # total_spoilers = 0.0
    # spoilers_length = 0.0
    # for i, sentence in enumerate([x[kTEXT_FIELD] for x in train]):
    #     if y_train[i] == True:
    #         # print sentence
    #         sentence = KeepWordsAndSpaces(sentence)
    #         sentence = word_tokenize(sentence)
    #         spoilers_length += len(sentence)
    #         total_spoilers += 1

    # print 'Average spoiler length in words', spoilers_length/total_spoilers

    # total_not_spoilers = 0.0
    # not_spoilers_length = 0.0
    # for i, sentence in enumerate([x[kTEXT_FIELD] for x in train]):
    #     if y_train[i] == False:
    #         # print sentence
    #         sentence = KeepWordsAndSpaces(sentence)
    #         sentence = word_tokenize(sentence)
    #         not_spoilers_length += len(sentence)
    #         total_not_spoilers += 1

    # print 'Average not spoiler length in words', not_spoilers_length/total_not_spoilers
