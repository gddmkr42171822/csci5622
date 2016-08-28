"""
Robert Werthman
CSCI 5622
Homework 1: KNN
"""
import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree


class Numbers:
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

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"            

        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        # Find the counts for each of labels found with the indices.
        label_counts = {}
        for i in item_indices:
            if self._y[i] in label_counts:
                label_counts[self._y[i]] += 1
            else:
                label_counts[self._y[i]] = 1

        # Find the most common label(s).
        most_common_label = []
        for key, value in label_counts.iteritems():
            if value == max(label_counts.values()):
                most_common_label.append(key)

        # Check if there is more than one majority label.
        # Return the median of the majority of the labels if there is.
        # Otherwise just return the one majority label.
        if len(most_common_label) > 1:
            return median(numpy.array(most_common_label))
        else:
            return most_common_label[0]

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed.

        # Find the k nearest neighbors and create a list of their indices
        # to index into the label training array (self._y).  
        # kdtree.query returns an array of arrays so you have to 
        # get the first array of that array
        # for the indices list.
        _, nearest_neighbors = self._kdtree.query(
            numpy.array(example).reshape(1, -1), k=self._k)

        return self.majority(nearest_neighbors[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            ii = yy
            jj = self.classify(xx)
            if ii in d:
                if jj in d[ii]:
                    d[ii][jj] += 1
                else:
                    d[ii][jj] = 1
            else:
                d[ii] = {jj: 1}
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
