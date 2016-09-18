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

    # Support vectors are given by the equation y_i(w^t*x_i+b) = 1.0
    # This means a training sample (x_i) lies on its respective support
    # vector boundary if y_i(w^t*x_i+b) = 1.0.
    for i, training_sample in enumerate(x):
      sample_margin = y[i]*(w.dot(training_sample) + b)
      if abs(sample_margin - 1.0) <= tolerance:
        support.add(i)
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION

    # There will be slack if the training sample is on the wrong side of its
    # correct support vector boundary given by its label.  The training sample
    # will also have a margin < 1.  This is because the vector support 
    # boundary is now define by the quation y_i(s^t*x+b) = 1 - slack.
    for i, training_sample in enumerate(x):
      sample_margin = y[i]*(w.dot(training_sample) + b)
      if sample_margin < 1:
        slack.add(i)
    return slack


