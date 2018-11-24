import unittest

from tests.testDocLoader import TestDocLoader
from tests.testTfidf import TestTfidf
from tests.testExample import test_example
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

    s.add(TestDocLoader)
    s.add(TestTfidf)
    s.add(test_example)
    s.add(TestVaeAlexAdam)
    
    return s.s

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
