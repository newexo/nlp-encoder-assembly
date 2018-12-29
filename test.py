import unittest

from tests.testBytelevel import TestBytelevel
from tests.testDocLoader import TestDocLoader
from tests.testExample import test_example
from tests.testMnistVae import TestMnistVae
from tests.testRandomInit import TestRandomInit
from tests.testTfidf import TestTfidf
from tests.testVae import TestVaeAlexAdam


class countsuite():
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d, %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = countsuite()

    s.add(TestBytelevel)
    s.add(TestDocLoader)
    s.add(test_example)
    s.add(TestMnistVae)
    s.add(TestRandomInit)
    s.add(TestTfidf)
    s.add(TestVaeAlexAdam)
    
    return s.s


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
