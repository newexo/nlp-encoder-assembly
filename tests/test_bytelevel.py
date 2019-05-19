import unittest
import numpy as np
from numpy.linalg import norm

import bytelevel


class TestBytelevel(unittest.TestCase):
    def setUp(self):
        self.r = np.random.RandomState(42)
        self.data = ['extraordinary series of adventure', 'in the South Seas and elsewhere  ']
        self.foobar = ["foo", "bar"]
        self.foobar_onehot = bytelevel.onehot(bytelevel.encode(self.foobar))
        self.foobar_prediction = 0.5 * self.foobar_onehot + 0.48 * self.r.rand(*self.foobar_onehot.shape) 

    def tearDown(self):
        pass

    def testEncode(self):
        expected = np.array([[101, 120, 116, 114,  97, 111, 114, 100, 105, 110,  97, 114, 121,
                            32, 115, 101, 114, 105, 101, 115,  32, 111, 102,  32,  97, 100,
                            118, 101, 110, 116, 117, 114, 101],
                        [105, 110,  32, 116, 104, 101,  32,  83, 111, 117, 116, 104,  32,
                            83, 101,  97, 115,  32,  97, 110, 100,  32, 101, 108, 115, 101,
                            119, 104, 101, 114, 101,  32,  32]], dtype=np.uint8)

        actual = bytelevel.encode(self.data)
        self.assertEqual(0, norm(expected - actual))

    def testOnehot(self):
        expected = np.zeros((2, 3, 256))
        expected[0, 0, 102] = 1
        expected[0, 1, 111] = 1
        expected[0, 2, 111] = 1
        expected[1, 0, 98] = 1
        expected[1, 1, 97] = 1
        expected[1, 2, 114] = 1

        actual = bytelevel.onehot(bytelevel.encode(self.foobar))
        self.assertEqual(0, norm(expected - actual))

    def testDecode(self):
        expected = self.data
        actual = bytelevel.decode(bytelevel.encode(self.data))
        self.assertEqual(expected, actual)

    def testPrediction2str(self):
        expected = self.foobar
        actual = bytelevel.prediction2str(self.foobar_prediction)
        self.assertEqual(expected, actual)
