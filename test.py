import unittest

import tests.test_example

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

    s.add(tests.test_example.test_example)
    
    return s.s

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
