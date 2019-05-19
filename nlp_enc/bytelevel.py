import numpy as np


def str2nparray(s):
    return np.frombuffer(bytearray(s.encode('utf-8')), dtype=np.uint8)


def encode(a):
    return np.array([str2nparray(row) for row in a])


def onehot(a, vocab_size=256):
    m, n = a.shape
    temp = np.zeros((m, n, vocab_size))
    temp[np.expand_dims(np.arange(m), axis=0).reshape(m, 1), np.repeat(np.array([np.arange(n)]), m, axis=0), a] = 1
    return temp


def nparray2str(a):
    return bytearray(a.astype(np.ubyte)).decode('utf-8')


def decode(a):
    return [nparray2str(row) for row in a]


def prediction2str(p):
    a = p.argmax(axis=2)
    return decode(a)


