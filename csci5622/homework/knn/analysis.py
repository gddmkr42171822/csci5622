from knn import *
import matplotlib.pyplot as plt

def question1():
  num_training_examples = []
  accuracy = []
  data = Numbers("../data/mnist.pkl.gz")

  for i in range(0, 10):
    num_training_examples.append(2**i)
    knn = Knearest(data.train_x[:2**i], data.train_y[:2**i], 1)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy.append(knn.accuracy(confusion))

  plt.plot(num_training_examples, accuracy)
  plt.xlabel('Number of training examples')
  plt.ylabel('Accuracy (decimal)')
  plt.title(
    'Relationship between the number of training examples and accuracy')
  plt.savefig('q1.png')

def question2():
  num_k_neighbors = []
  accuracy = []
  data = Numbers("../data/mnist.pkl.gz")

  for k in range(1, 15):
    num_k_neighbors.append(k)
    knn = Knearest(data.train_x[:500], data.train_y[:500], k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy.append(knn.accuracy(confusion))

  plt.plot(num_k_neighbors, accuracy)
  plt.xlabel('Number of k-Nearest Neighbors')
  plt.ylabel('Accuracy (decimal)')
  plt.title(
    'Relationship between k-Nearest Neighbors and accuracy')
  plt.savefig('q2.png')


def main():
  question1()
  question2()
  

if __name__ == '__main__':
    main()