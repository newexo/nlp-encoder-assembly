import unittest
import docLoader

class TestDocLoader(unittest.TestCase):
    def setUp(self):
        self.seq = list(range(10))

    def tearDown(self):
        pass

    def testLoadDocs(self):
        docs = docLoader.loadDocs('testData')
        self.assertEqual(3, len(docs))
        self.assertEqual(26, len(docs[0]))
        self.assertEqual('12/11/89\n', docs[0][0])

    def testCollectLines(self):
        testRomeo = ['SAMPSON.\n',
             'A dog of the house of Montague moves me.\n',
             '\n',
             'GREGORY.\n',
             'To move is to stir; and to be valiant is to stand: therefore, if thou\n',
             'art moved, thou runn’st away.\n',
             '\n',
             'SAMPSON.\n',
             'A dog of that house shall move me to stand.\n',
             'I will take the wall of any man or maid of Montague’s.\n',
             '\n',
             'GREGORY.\n',
             'That shows thee a weak slave, for the weakest goes to the wall.\n',
             '\n']
        speakers, spoken = docLoader.collectLines(testRomeo)

        expected = ['SAMPSON', 'GREGORY', 'SAMPSON', 'GREGORY']
        self.assertEqual(expected, speakers)

        expected = ['\n',
              'To move is to stir; and to be valiant is to stand: therefore, if thou\n',
              'art moved, thou runn’st away.\n',
              '\n',
              '\n',
              'That shows thee a weak slave, for the weakest goes to the wall.\n',
              '\n']
        self.assertEqual(expected, spoken)

    # def testFoo(self):
    #     self.assertTrue(False, "incomplete test")

