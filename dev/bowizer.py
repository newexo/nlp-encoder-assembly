import nltk
from nltk import word_tokenize
import collections
import tfidf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import bytelevel


class BOWizer():
    def __init__(self, extendVocabList, td, vocab):
        self.extendVocabList = extendVocabList
        self.td = td
        self.vocab = vocab
        
    def bow(self, raw_text):
        tokenized_text = word_tokenize(raw_text)
        unked = [tfidf.toUnk(x, self.vocab) for x in list(tokenized_text)]
        tokenlist = tfidf.toTokenList(unked, self.td)
        bow = tfidf.toBOW(tokenlist, len(self.extendVocabList))
        
        bow = bow / np.sum(bow)
        return bow


def get_vocab(tokenized_text, vocab_size, unk='unk'):
    C = collections.Counter(tokenized_text)
    sort_ = C.most_common()
    tokens = sort_[:vocab_size]
    vocab = [t for t, _ in tokens]
    extendVocabList, td = tfidf.getTokenDict(vocab, unk=unk)
    return extendVocabList, td, set(vocab)


def tokenize_docs(list_of_docs):
    tokens = []
    for doc in list_of_docs:
        tokens.extend(word_tokenize(doc))
    return tokens


def make_bowizer(corpus, vocabsize):
    extendVocabList, td, vocab = get_vocab(tokenize_docs(corpus), vocabsize)
    bow = BOWizer(extendVocabList, td, vocab)
    return bow


class TokenMaker(object):
    def __init__(self, corpus, vocab_size, should_lower=True, unk='unk'):
        self.unk = unk
        self.should_lower = should_lower
        tokenized_text = tokenize_docs(corpus)
        tokenized_text = self.canonical(tokenized_text)
        self.vocab_size = vocab_size
        self.extendVocabList, self.td, self.vocab = get_vocab(tokenized_text, vocab_size, unk=self.unk)
        
    def canonical(self, words):
        if self.should_lower:
            return [word.lower() for word in words]
        return words
    
    def vector(self, text):
        words = word_tokenize(text)
        words = [tfidf.toUnk(word, self.vocab, self.unk) for word in words]
        return [self.td[word] for word in words]
    
    def x(self, lines, maxlen=None):
        vectors = [self.vector(line) for line in lines]
        return pad_sequences(vectors, maxlen=maxlen)
    
    def y(self, x):
        return bytelevel.onehot(x, self.vocab_size + 1)


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


def test():
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')[:1000]
    grail = nltk.corpus.webtext.raw('grail.txt')[:1000]
    corpus = [grail, emma]

    bowizer = make_bowizer(corpus, 10)
    print(bowizer.bow(emma[100:200]))


if __name__ == "__main__":
    test()
