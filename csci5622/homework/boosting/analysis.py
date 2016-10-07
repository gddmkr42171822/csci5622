from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from boost import AdaBoost
from boost import FoursAndNines
import numpy as np 
import matplotlib.pyplot as plt

def Question2():
  depths = [1, 2, 3, 4]
  num_learners = [1, 10, 50, 100, 300]

  data = FoursAndNines("../data/mnist.pkl.gz")
  print 'Size of training set', data.x_train.shape[0]
  print 'Size of validation set', data.x_valid.shape[0]
  for depth in depths:
    print 'Tree depth', depth
    accuracy = []
    clf = AdaBoost(n_learners=max(num_learners), base=DecisionTreeClassifier(
      max_depth=depth, criterion="entropy"))

    print 'Fitting classifier...'
    clf.fit(data.x_train, data.y_train)

    print 'Calculating scores...'
    temp_learners = clf.learners
    for num_learner in num_learners:
      clf.learners = temp_learners[:num_learner]
      print 'Number of learners', len(clf.learners)
      # score = clf.score(data.x_valid, data.y_valid)
      score = clf.score(data.x_train, data.y_train)
      print 'Score at iteration %d with depth %d is %f' % (
        num_learner, depth, score)
      accuracy.append(score)

    plt.plot(num_learners, accuracy, 'o', label=str(depth))

  plt.title('Accuracy of decision tree boosting vs the\nnumber of weak learners at a certain depth with training set')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of weak learners')
  plt.legend(title='Depth', numpoints=1)
  plt.savefig('q2_train_accuracy.png')

def Question3():
  depths = [1, 2, 3, 4]
  num_learners = [1, 10, 50, 100, 300]

  data = FoursAndNines("../data/mnist.pkl.gz")
  print 'Size of training set', data.x_train.shape[0]
  print 'Size of validation set', data.x_valid.shape[0]
  accuracy = []
  clf = AdaBoost(n_learners=max(num_learners), base=MultinomialNB())

  print 'Fitting classifier...'
  clf.fit(data.x_train, data.y_train)

  print 'Calculating scores...'
  temp_learners = clf.learners
  for num_learner in num_learners:
    clf.learners = temp_learners[:num_learner]
    print 'Number of learners', len(clf.learners)
    score = clf.score(data.x_valid, data.y_valid)
    print 'Score at iteration %d is %f' % (num_learner, score)
    accuracy.append(score)

  plt.plot(num_learners, accuracy, 'o')

  plt.title(
    'Accuracy of Multinomial Naive Bayes\nvs the number of weak learners')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of weak learners')
  plt.savefig('q3.png')

if __name__ == '__main__':
  Question2()