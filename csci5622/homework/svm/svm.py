import numpy as np 

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w. 
    """

    w = np.zeros(len(x[0]))
    # TODO: IMPLEMENT THIS FUNCTION
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices 
    of all of the support vectors
    """

    support = set()
    # TODO: IMPLEMENT THIS FUNCTION

    positive_support_vector_margin = 1.0
    negative_support_vector_margin = -1.0
    # Use the decision boundary function (w^t*x_i+b) to find the number
    # of margins each training sample is from the decision boundary.
    for i, training_sample in enumerate(x):
      sample_margin = w.dot(training_sample) + b
      # If the training sample has a positive label (1) the support vector
      # boundary is the equation w^t*x_i+b = margin. 
      # If the margin = 1, the the training
      # sample would be on the positive support vector boundary and be a
      # support vector.
      if (y[i] == 1):
        if abs(sample_margin - positive_support_vector_margin) <= tolerance:
          support.add(i)
      # If the training sample has a negative label (-1) the support vector 
      # margin = -1. If the training sample produces that margin when 
      # pugged into the decision boundary function, the sample
      # would be on the negative support vector boundary and be a support 
      # vector for the negative class.
      elif y[i] == -1:
        if abs(sample_margin - negative_support_vector_margin) <= tolerance:
          support.add(i)
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    return slack


