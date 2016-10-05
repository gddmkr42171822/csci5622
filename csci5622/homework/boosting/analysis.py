from sklearn.tree import DecisionTreeClassifier
from boost import AdaBoost
from boost import FoursAndNines
import numpy as np 
import matplotlib.pyplot as plt

def Question2():
  depths = [1, 2]
  num_learners = [10, 11, 12, 13, 14]

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
    print 'Calculating staged scores...'
    staged_scores = clf.staged_score(data.x_valid, data.y_valid)
    for learner in num_learners:
      print 'Staged score at iteration %d with depth %d is %f' % (
        learner, depth, staged_scores[learner - 1])
      accuracy.append(staged_scores[learner - 1])

    plt.plot(num_learners, accuracy, 'o', label=str(depth))

  plt.title('Accuracy of decision tree boosting\nvs the number of weak learners at a certain depth')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of weak learners')
  # Fix the tick marks on the plot so they are whole numbers and space 
  # on the end points
  extraticks = [min(num_learners) - 1, max(num_learners) + 1]
  plt.xticks(num_learners)
  plt.xticks(list(plt.xticks()[0]) + extraticks)
  plt.legend(title='Depth', numpoints=1)
  plt.savefig('q2.png')

if __name__ == '__main__':
  Question2()