import nltk
from nltk import word_tokenize
import collections
import tfidf
import numpy as np
np.random.seed(1234)


class BOWizer():
    def __init__(self, extendVocabList, td, vocab):
        self.extendVocabList =extendVocabList
        self.td =td
        self.vocab=  vocab
        
    def bow(self, raw_text):
        tokenized_text = word_tokenize(raw_text)
        unked = [tfidf.toUnk(x, self.vocab) for x in list(tokenized_text)]
        #print(unked)
        tokenlist = tfidf.toTokenList(unked, self.td)
        #print(tokenlist)
        bow = tfidf.toBOW(tokenlist,len(self.extendVocabList))
        
        bow = bow/ np.sum(bow)
        return bow
        
        
        
        
def get_vocab(tokenized_text, vocab_size):
    
    C = collections.Counter(tokenized_text)
    sort_ = C.most_common()
    tokens = sort_[:vocab_size]
    vocab = {t for t, _ in tokens}  
    extendVocabList, td =  tfidf.getTokenDict(vocab)
    return extendVocabList, td, vocab

def tokenize_docs(list_of_docs):
    tokens = []
    for doc in list_of_docs:
        tokens.extend(word_tokenize(doc))
    return tokens

def make_bowizer(corpus, vocabsize):
    extendVocabList, td, vocab = get_vocab(tokenize_docs(corpus), vocabsize)
    bow = BOWizer(extendVocabList, td, vocab)
    return bow


def test():
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')[:1000]
    grail = nltk.corpus.webtext.raw('grail.txt')[:1000]
    corpus = [grail, emma]

    bowizer = make_bowizer(corpus, 10)
    print(bowizer.bow(emma[100:200]))


if __name__ == "__main__":
    test()


