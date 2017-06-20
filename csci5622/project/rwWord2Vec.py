"""
TODO: Add stopwords from nltk or sklearn.
TODO: Add stemming from nltk (SnowballStemmer).
TODO: Try TfIdfVectorizer from sklearn.
"""
import csv
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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from gensim.models.word2vec import Word2Vec
import gensim



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

def DecisionTreeStump(X, y):
  clf = DecisionTreeClassifier(max_depth=1)
  print 'Decision Tree Stump score', cross_val_score(clf, X, y).mean()

def AdaBoostDecisionTree(X, y):
  clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
  print 'AdaBoosted Decision Tree score', cross_val_score(clf, X, y).mean()

def PerceptronPrediction(X, y):
  clf = Perceptron()
  print 'Perceptron score', cross_val_score(clf, X, y).mean()

def AdaboostPerceptron(X, y):
  clf = AdaBoostClassifier(Perceptron(), algorithm='SAMME')
  print 'AdaBoosted Perceptron score', cross_val_score(clf, X, y).mean()


def word2vecClassifier(X,y):
  #X=PreprocessSamples(X)
  """
  stop_words = stopwords.words('english')
  X_modified = []
  for line in X:
    line = KeepWordsSpaces(line)
    line = line.split()
    line = RemoveStopWords(stop_words, line)
    line = ' '.join(line)
    X_modified.append(line)
  X=X_modified[:]
  """


 
  sentences = []
  for entry in X:
    sentence=[]
    for word in entry.split():
      sentence.append(word)
    sentences.append(sentence)
  X=sentences[:]
 
  model = Word2Vec(X, size=200, window=5, min_count=1, workers=4)
  vectorized=[]
  for line in X:
    #print line
    temp=np.zeros(200)
    counter=0.
    for word in line:
      temp=np.add(temp,np.array(model[word]))
      counter+=1.
    temp=np.divide(temp,counter)
    vectorized.append(temp)
  #vectorized=vectorizer.fit_transform(vectorized)
  #clf = MultinomialNB()
  #print 'Multinomial Naive Bayes score word2vec', cross_val_score(clf, vectorized, y).mean()
  #for index in range(0,len(X)):
    #print X[index]
    #temp=(str(vectorized[index]))
    #X[index].append(temp)
    #print X[index]
  #X = vectorizer.fit_transform(X)
  print "word2Vec"
  #MultinomialNaiveBayes(X, y)
  DecisionTreeStump(vectorized, y)
  AdaBoostDecisionTree(vectorized, y)
  PerceptronPrediction(vectorized, y)
  AdaboostPerceptron(vectorized, y)
  print "\n"

def main():
  
  vectorizer = CountVectorizer()
  # vectorizer = TfidfVectorizer()

  # Create a list of the relatives paths to all of the csv files.
  samples = []
  for f in listdir('csv_data/'):
    f = join('csv_data/', f)
    if isfile(f):
      samples.append(f)

  X, y = GetSamples(samples)
  # X = PreprocessSamples(X)

  X, y = shuffle(X, y, random_state=random.randint(1, 100))
  word2vecClassifier(X,y)
  X = vectorizer.fit_transform(X)

  MultinomialNaiveBayes(X, y)
  DecisionTreeStump(X, y)
  AdaBoostDecisionTree(X, y)
  PerceptronPrediction(X, y)
  AdaboostPerceptron(X, y)
  


if __name__ == '__main__':
  main()