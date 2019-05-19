import bytelevel
from keras.preprocessing.sequence import pad_sequences

def make_chunks(data, count=100):
    n = int(len(data) / float(count))
    return [data[i:i+n] for i in range(0, len(data), n)]

def make_train_test(chunks, train_ratio=0.9):
    n = int(len(chunks) * train_ratio)
    train = ''.join(chunks[:n])
    test = ''.join(chunks[n:])
    return train, test

class SlicedData(object):
    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len
        x = bytelevel.encode(text)
        self.x = pad_sequences(x, max_len)
        self.y = bytelevel.onehot(self.x)
        
    @staticmethod
    def Random(train_text, test_text, max_len, n, r):
        def random_slice(data):
            i = r.randint(len(data) - max_len)
            return data[i : i + max_len]
        train_slices = [random_slice(train_text) for _ in range(n)]
        test_slices = [random_slice(test_text) for _ in range(int(0.1 * n))]

        return SlicedData(train_slices, max_len), SlicedData(test_slices, max_len)