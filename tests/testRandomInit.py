import unittest

import numpy as np
from numpy.linalg import norm

from . import random_init

class TestRandomInit(unittest.TestCase):
    def setUp(self):
        class MockModel(object):
            def get_weights(self):
                return [np.zeros((13,)), np.zeros((5, 7))]

        self.model = MockModel()

        w13 = np.array([0.04240832, 0.1076472 , 0.08288199, 0.06778473, 0.01766563,
                        0.0176629 , 0.00657667, 0.09807514, 0.06806287, 0.08017343,
                        0.00233074, 0.10982067, 0.09425557])
        w5_7 = np.array([[0.0250244 , 0.02142828, 0.02161443, 0.03585529, 0.06184314,
                        0.05090521, 0.03432168],
                        [0.07210756, 0.01643951, 0.03442958, 0.04317616, 0.05374836,
                        0.09253387, 0.02353178],
                        [0.06060311, 0.06981673, 0.00547423, 0.07159985, 0.02009646,
                        0.0076664 , 0.11182723],
                        [0.11380083, 0.09527054, 0.03589908, 0.01151077, 0.08063764,
                        0.05187247, 0.01438234],
                        [0.05835716, 0.00405273, 0.10716444, 0.03049751, 0.078079  ,
                        0.0367355 , 0.0612906 ]])

        self.expected_weights = [w13, w5_7]

    def tearDown(self):
        pass

    def test_scale(self):
        expected = 0.11785113019775793
        actual = random_init.scale((5, 7))
        self.assertAlmostEqual(expected, actual)
        
    def test_get_shapes(self):
        actual = random_init.get_shapes(self.model)
        expected = [(13,), (5, 7)]
        self.assertEqual(expected, actual)
        
    def test_random_from_shapes(self):
        r = np.random.RandomState(42)
        actual = random_init.random_from_shapes(r, [(13,), (5, 7)])
        self.assertEqual(2, len(actual))

        expected = self.expected_weights
        for i in range(2):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]))
        
    def test_random_w(self):
        r = np.random.RandomState(42)
        actual = random_init.random_w(r, self.model)
        self.assertEqual(2, len(actual))

        expected = self.expected_weights
        for i in range(2):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]))
