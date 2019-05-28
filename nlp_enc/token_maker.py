import collections

from keras.preprocessing.sequence import pad_sequences
from nlp_enc import tfidf, bytelevel
from nltk import word_tokenize


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