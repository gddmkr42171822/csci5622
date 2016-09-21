import argparse
import numpy as np 

from sklearn.svm import SVC
from sklearn import cross_validation
import matplotlib.pyplot as plt
from distutils.command.build_scripts import first_line_re

class ThreesAndEights:
  """
  Class to store MNIST data
  """

  def __init__(self, location):
    # You shouldn't have to modify this class, but you can if
    # you'd like.
    
    import cPickle, gzip
    
    # Load the dataset
    f = gzip.open(location, 'rb')
    
    train_set, valid_set, test_set = cPickle.load(f)
    
    self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
    self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]
    
    shuff = np.arange(self.x_train.shape[0])
    np.random.shuffle(shuff)
    self.x_train = self.x_train[shuff,:]
    self.y_train = self.y_train[shuff]
    
    self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
    self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
    
    self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
    self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]
    
    f.close()
    
  def cross_validate(self, C=None, kernel=None):
    # Initialize the support vector machine classifier
    svm = None
    if C is not None and kernel is not None:
      svm = SVC(C=C, kernel=kernel)
    else:
      svm = SVC()
    print svm.get_params()
    
    # Combines all of the samples and labels from the training, test, 
    # and validation sets
    self.samples = self.x_train
    self.labels = self.y_train
    
#     self.samples = np.vstack((self.samples, self.x_test))
#     self.samples = np.vstack((self.samples, self.x_valid))
#     
#     self.labels = np.append(self.labels, self.y_test)
#     self.labels = np.append(self.labels, self.y_valid)

    # Do cross validation and look at the mean of the scores of the each
    # of the data splits as the accuracy
    folds = 3
    scores = cross_validation.cross_val_score(
      svm, self.samples, self.labels, cv=folds, n_jobs=-1)
    return scores.mean()

  def predict(self, sample, C=None, kernel=None):
    svm = None
    if C is not None and kernel is not None:
      svm = SVC(C=C, kernel=kernel)
    else:
      svm = SVC()
    print svm.get_params()
    svm.fit(self.x_train, self.y_train)
    
#     print 'Prediction: %d\n' % svm.predict(sample.reshape(1,-1))
    return svm
    
  def plot_accuracy(self):
    print self.accuracy
    for kernel in self.kernels:
      plt.plot(self.C, self.accuracy[kernel], label=kernel)
    
    
    plt.title(
      'Accuracy of svm prediction with different values of C for the kernels: %s' % (', '.join(self.kernels)))
    plt.xlabel('Values of C')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    plt.savefig('performance_evalation.png')

def mnist_digit_show(flatimage, outname=None):
  
  image = np.reshape(flatimage, (-1,28))
  
  plt.matshow(image, cmap=plt.cm.binary)
  plt.xticks([])
  plt.yticks([])
  if outname: 
    plt.savefig(outname)
  else:
    plt.show()

  
def question1(data):
  data.C = [.1, 1, 10, 100, 1000]
  data.kernels = ['linear', 'rbf']
  data.accuracy = {}
   
  for kernel in data.kernels:
    for C in data.C:
      accuracy = data.cross_validate(C=C, kernel=kernel)
      if kernel in data.accuracy:
        data.accuracy[kernel].append(accuracy)
      else:
        data.accuracy[kernel] = [accuracy]
      print 'c %d, kernel %s, accuracy %f' % (
        C, kernel, accuracy)
   
  data.plot_accuracy()


def question2(data):
  sample = data.x_valid[0]
  svm = data.predict(sample=sample, C=1, kernel='linear')

  # Get support vectors for the first class and second class
  first_class_support_vectors = svm.support_vectors_[0:3]
  second_class_support_vectors = svm.support_vectors_[
    svm.n_support_[0]:svm.n_support_[0]+3]
  
  i = 0
  for sv in first_class_support_vectors:
    if i < 3:
      mnist_digit_show(sv, '3_sv%d.png' % i)
    else:
      break
    i += 1

  i = 0
  for sv in second_class_support_vectors:
    if i < 3:
      mnist_digit_show(sv, '8_sv%d.png' % i)
    else:
      break
    i += 1


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='SVM classifier options')
  parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
  args = parser.parse_args()
  
  data = ThreesAndEights("../data/mnist.pkl.gz")

  """
  data.x_test is a matrix where one row and all of its columns 
  represent a number like 3 or 8.
    example: data.x_test[0,:] is the first row and all of its columns
             - when this data is drawn it produces the number 3
  
  data.y_test is an array in which each index is a label for the row of 
  data.x_test
    example: data.y_test[0] is the label for data.x_test[0,:]
             data.y_test[1] is the label for data.x_test[1,:]
  """
  question1(data)
#   question2(data)