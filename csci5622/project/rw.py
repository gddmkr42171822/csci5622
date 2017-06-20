import csv
import random
import re
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.pipeline import FeatureUnion

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag

# Prevent the csv reader from throwing an error about the size of the 
# field it is reading in.
csv.field_size_limit(sys.maxsize)

class POSCountTokenizer(object):
  def __call__(self, text):
      words = text.split()
      words_and_pos_tags = pos_tag(words)
      # Return a list of the part of speech for each word in the sentence.
      return [word_and_pos[1] for word_and_pos in words_and_pos_tags]


def GetSamples(files):
  X = []
  y = []
  for f in files:
    if f.endswith('.csv'):
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

def PreprocessSamples(X, ws, sw, st):
  stop_words = stopwords.words('english')
  stemmer = SnowballStemmer('english')
  X_modified = []
  # Preprocess the training examples: remove stop words, stem the words, etc.
  for line in X:
    if ws: line = KeepWordsSpaces(line)
    # Tokenize the line into a list of words.
    line = line.split()
    if sw: line = RemoveStopWords(stop_words, line)
    if st: line = StemText(stemmer, line)
    # Rejoin the list of tokens in to a single string.
    line = ' '.join(line)
    X_modified.append(line)
  return np.array(X_modified)

def MultinomialNaiveBayes(X, y):
  clf = MultinomialNB()
  print 'Multinomial Naive Bayes score', cross_val_score(clf, X, y).mean()

def PerceptronPrediction(X, y):
  clf = Perceptron()
  print 'Perceptron score', cross_val_score(clf, X, y).mean()

def main():
  
  bag_of_words_vectorizer = CountVectorizer()
  pos_vectorizer = CountVectorizer(tokenizer=POSCountTokenizer())

  # Create a list of the relatives paths to all of the csv files.
  samples = []
  for f in listdir('csv_data/'):
    f = join('csv_data/', f)
    if isfile(f):
      samples.append(f)

  X, y = GetSamples(samples)
  print 'Number of training samples', len(X)
  X = PreprocessSamples(X, True, True, True)

  fu = FeatureUnion([
    ('bag_of_words', bag_of_words_vectorizer),
    ('pos_count', pos_vectorizer)])

  X, y = shuffle(X, y, random_state=random.randint(1, 100))
  X = fu.fit_transform(X)
  # X = bag_of_words_vectorizer.fit_transform(X)

  MultinomialNaiveBayes(X, y)
  PerceptronPrediction(X, y)


if __name__ == '__main__':
  main()