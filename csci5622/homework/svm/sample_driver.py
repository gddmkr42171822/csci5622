import argparse
import numpy as np 

from sklearn.svm import SVC

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

  def predict(self, C=None, kernel=None):
    svm = None
    if C and kernel:
      svm = SVC(C=C, kernel=kernel)
    else:
      svm = SVC()
    
    svm.fit(self.x_train, self.y_train)
    
    print svm.predict(self.x_test[0,:])
    print svm.score(self.x_test[0,:], self.y_test[0])

def mnist_digit_show(flatimage, outname=None):

  import matplotlib.pyplot as plt
  
  image = np.reshape(flatimage, (-1,28))
  
  plt.matshow(image, cmap=plt.cm.binary)
  plt.xticks([])
  plt.yticks([])
  if outname: 
    plt.savefig(outname)
  else:
    plt.show()
  
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
  
  data.y_test is an array in which each index is a label for the row of 
  data.x_test
    example: data.y_test[0] is the label for data.x_test[0,:]
             data.y_test[1] is the label for data.x_test[1,:]
  """
  
  
  print data.x_test[0,:]
  print data.x_test[1,:]
  
  data.predict()
  # -----------------------------------
  # Plotting Examples 
  # -----------------------------------
  
  # Display in on screen
  mnist_digit_show(data.x_test[0,:])  
  # 	mnist_digit_show(data.x_train[ 0,:])
  
  # Plot image to file 
  # 	mnist_digit_show(data.x_train[1,:], "mnistfig.png")









