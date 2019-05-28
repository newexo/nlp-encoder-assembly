from nlp_enc.token_maker import TokenMaker
from nltk import word_tokenize
from nlp_enc import tfidf
import numpy as np


class BOWizer():
    def __init__(self, tm: TokenMaker):
        self.tm = tm
        
    def bow(self, raw_text):
        tokenized_text = word_tokenize(raw_text)
        unked = [tfidf.toUnk(x, self.tm.vocab) for x in list(tokenized_text)]
        tokenlist = tfidf.toTokenList(unked, self.tm.td)
        bow = tfidf.toBOW(tokenlist, len(self.tm.extendVocabList))
        
        bow = bow / np.sum(bow)
        return bow


def make_bowizer(corpus, vocabsize):
    tm = TokenMaker(corpus, vocabsize)
    bow = BOWizer(tm)
    return bow


class SlicedWordData(object):
    def __init__(self, lines, maxlen, tokenmaker):
        self.line = lines
        self.maxlen = maxlen
        self.tokenmaker = tokenmaker
        self.x = self.tokenmaker.x(lines, maxlen=self.maxlen)
        self.y = self.tokenmaker.y(self.x)
        
    @staticmethod
    def Random(train_text, test_text, linelen, maxlen, n, r, tokenmaker):
        def random_slice(data):
            i = r.randint(len(data) - linelen)
            return data[i : i + linelen]
        train_slices = [random_slice(train_text) for _ in range(n)]
        test_slices = [random_slice(test_text) for _ in range(int(0.1 * n))]

        return SlicedWordData(train_slices, maxlen, tokenmaker), SlicedWordData(test_slices, maxlen, tokenmaker)
