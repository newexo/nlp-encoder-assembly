import unittest
import numpy as np
from numpy.linalg import norm

import bytelevel

class TestBytelevel(unittest.TestCase):
    def setUp(self):
        self.data = ['extraordinary series of adventure', 'in the South Seas and elsewhere  ']

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

    # def testOnehot(self):
    #     self.assertTrue(False)

    # def testDecode(self):
    #     self.assertTrue(False)

    # def testPrediction2(self):
    #     self.assertTrue(False)
