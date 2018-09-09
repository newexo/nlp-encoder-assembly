import unittest
import tfidf

class TestTfidf(unittest.TestCase):
    def setUp(self):
        self.seq = list(range(10))

    def tearDown(self):
        pass

    def testReadStopList(self):
        testStops = tfidf.readStopList()
        self.assertEqual('i', testStops[10])
        #self.assertEqual(false, "Incomplete test")

    def testCollectPhrases(self):
        testStops = tfidf.readStopList()
        testSentences = [['extraordinary series of adventure in the South Seas and elsewhere, of which an account is given in the following pages, accident threw me into the society of several gentlemen in Richmond, Va.'], 
                      ['were constantly urging it upon me, as a duty, to give my narrative to of which were of a nature altogether private, and concern no person but myself; others not so much so.'], 
                      ['unavoidable exaggeration to which all of us are prone when detailing events which have had powerful influence in exciting the imaginative faculties.'], 
                      ['in my veracity--the probability being that the public at large would one of the principal causes which prevented me from complying with the suggestions of my advisers.']]
        testPhrases = tfidf.collectPhrases(testSentences, testStops)

        self.assertEqual('South_Seas', testPhrases[2])

    def testCollectWords(self):
        testSentences = [['extraordinary series of adventure in the South Seas and elsewhere, of which an account is given in the following pages, accident threw me into the society of several gentlemen in Richmond, Va.'], 
                      ['were constantly urging it upon me, as a duty, to give my narrative to of which were of a nature altogether private, and concern no person but myself; others not so much so.'], 
                      ['unavoidable exaggeration to which all of us are prone when detailing events which have had powerful influence in exciting the imaginative faculties.'], 
                      ['in my veracity--the probability being that the public at large would one of the principal causes which prevented me from complying with the suggestions of my advisers.']]
        testWords = tfidf.collectWords(testSentences)
        self.assertEqual('extraordinary', testWords[0])
        self.assertFalse(False, "Incomplete test")

    def testWordFreq(self):
        testWords = ['extraordinary', 'series', 'of', 'adventure', 'in', 'the', 'South', 'Seas', 'and', 'elsewhere']
        testWFreqs = tfidf.wordFreq(testWords)
        self.assertEqual(2, testWFreqs['in'])
        self.assertEqual(1, testWFreqs['adventure']) 
        #self.assertFalse(False, "Incomplete test")        