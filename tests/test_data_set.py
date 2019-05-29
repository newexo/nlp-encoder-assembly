import unittest
import numpy as np

from nlp_enc import data_set


class TestDataSet(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_make_chunks(self):
        data = list(range(17))
        actual = data_set.make_chunks(data, count=4)
        expected = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16]]
        self.assertEqual(expected, actual)

    def test_make_train_test(self):
        r = np.random.RandomState(42)
        data = 'qwertyuiopasdfghjklzxcvbnm'
        chunks = data_set.make_chunks(data, count=4)
        r.shuffle(chunks)
        train, test = data_set.make_train_test(chunks, train_ratio=0.8)

        expected = 'uiopasnmdfghjkqwerty'
        actual = train
        self.assertEqual(expected, actual)

        expected = 'lzxcvb'
        actual = test
        self.assertEqual(expected, actual)
