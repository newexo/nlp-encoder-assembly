import unittest

from tests.test_bowizer import TestBowizer
from tests.test_bytelevel import TestBytelevel
from tests.test_doc_loader import TestDocLoader
from tests.test_example import TestExample
from tests.test_hyper import TestHyper
from tests.test_MNIST_VAE import TestMnistVae
from tests.test_random_init import TestRandomInit
from tests.test_text_decoder import TestTextDecoder
from tests.test_text_encoder import TestTextEncoder
from tests.test_TFIDF import TestTfidf
from tests.test_VAE import TestVaeAlexAdam


class CountSuite:
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = CountSuite()

    s.add(TestBowizer)
    s.add(TestBytelevel)
    s.add(TestDocLoader)
    s.add(TestExample)
    s.add(TestHyper)
    s.add(TestMnistVae)
    s.add(TestRandomInit)
    s.add(TestTextDecoder)
    s.add(TestTextEncoder)
    s.add(TestTfidf)
    s.add(TestVaeAlexAdam)
    
    return s.s


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
