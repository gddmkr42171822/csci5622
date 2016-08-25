import unittest

from numpy import array

from knn import *

import mock

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                        [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
        self.y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1])
        self.knn = {}
        for ii in [1, 2, 3]:
            self.knn[ii] = Knearest(self.x, self.y, ii)

        self.queries = array([[1, 5], [0, 3], [6, 1], [6, 4]])

    def test1(self):
        self.assertAlmostEqual(self.knn[1].classify(self.queries[0]), 1)
        self.assertAlmostEqual(self.knn[1].classify(self.queries[1]), -1)
        self.assertAlmostEqual(self.knn[1].classify(self.queries[2]), 1)
        self.assertAlmostEqual(self.knn[1].classify(self.queries[3]), -1)

    def test2(self):
        self.assertAlmostEqual(self.knn[2].classify(self.queries[0]), 1)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[1]), 0)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[2]), 0)
        self.assertAlmostEqual(self.knn[2].classify(self.queries[3]), -1)

    def test3(self):
        self.assertAlmostEqual(self.knn[3].classify(self.queries[0]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[1]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[2]), 1)
        self.assertAlmostEqual(self.knn[3].classify(self.queries[3]), -1)

    def testMajority(self):
        item_indices = [1, 2, 8]
        return_value = self.knn[3].majority(item_indices)
        self.assertEqual(+1, return_value)

    def testMajorityWithMedianResult(self):
        item_indices = [1, 8]
        return_value = self.knn[2].majority(item_indices)
        self.assertEqual(0, return_value)

    @mock.patch.object(Knearest, 'majority')
    def testClassifyMockMajorityWithOneNN(self, mock_majority):
        query = [2, 0]
        x = array([[2, 0], [4,1]])
        y = array([+1, -1])
        knn = Knearest(x, y, 1)
        knn.classify(query)
        mock_majority.assert_called_once_with(array([0]))

    @mock.patch.object(Knearest, 'majority')
    def testClassifyMockMajorityWithTwoNN(self, mock_majority):
        query = [2, 0]
        x = array([[2, 0], [4, 1], [2, 1]])
        y = array([+1, +1, -1])
        knn = Knearest(x, y, 2)
        knn.classify(query)
        numpy.testing.assert_array_equal(
            array([0, 2]), mock_majority.call_args[0][0])

    def testClassifyWithtwoNN(self):
        query = [2, 0]
        x = array([[2, 0], [4, 1], [2, 1]])
        y = array([+1, +1, -1])
        knn = Knearest(x, y, 2)
        actual_label = knn.classify(query)
        self.assertEqual(0, actual_label)

    def testConfusionMatrixNoIncorrectLabels(self):
        expected_d = defaultdict(dict)
        expected_d[-1] = {-1: +1}
        expected_d[+1] = {+1: +1}
        test_x = array([[1, 1], [4, 1]])
        test_y = array([-1, +1])
        x = array([[2, 0], [4, 1], [2, 1]])
        y = array([+1, +1, -1])
        knn = Knearest(x, y, 1)
        actual_d = knn.confusion_matrix(test_x, test_y)
        self.assertEqual(expected_d, actual_d)

    def testConfustionMatrixWithIncorrectLabels(self):
        expected_d = defaultdict(dict)
        expected_d[+1] = {+1: 1, -1: 1}
        test_x = array([[1, 1], [4, 1]])
        test_y = array([+1, +1])
        x = array([[2, 0], [4, 1], [2, 1]])
        y = array([+1, +1, -1])
        knn = Knearest(x, y, 1)
        actual_d = knn.confusion_matrix(test_x, test_y)
        self.assertEqual(expected_d, actual_d)

if __name__ == '__main__':
    unittest.main()
