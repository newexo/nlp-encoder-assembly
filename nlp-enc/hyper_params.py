from keras.layers import Embedding, Conv1D, Input, GRU, LSTM, Bidirectional, Dense, UpSampling1D, Dropout, TimeDistributed, RepeatVector
from keras.optimizers import Adam


def name_prefix(prefix):
    if prefix is None or not len(prefix.strip()):
        return ''
    return '%s_' % prefix


class Hyper(object):
    def __init__(self, 
            lr=0.001,
            batch_size=10,
            epochs=3):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def random(r):
        lr = r.choice([0.01 * (10 ** -(0.5 * i)) for i in range(6)])
        lr = float(lr)
        batch_size = r.choice([5, 10, 20, 50])
        batch_size = int(batch_size)
        epochs = 3
        return Hyper(lr, batch_size, epochs)

    def display(self):
        print("learing rate=%f" % self.lr)
        print("batch size=%d" % self.batch_size)
        print("training epochs=%d" % self.epochs)

    def make_optimizer(self):
        return Adam(lr=self.lr)


class EmbeddingHyper(object):
    def __init__(self, 
            vocab_size=256, 
            embedding_dim=64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    @property
    def default_name(self):
        return 'embedder'

    @staticmethod
    def random(r):
        embedding_dim = r.choice([2 ** i for i in range(6, 10)])
        embedding_dim = int(embedding_dim)
        return EmbeddingHyper(256, embedding_dim)
        
    def display(self):
        print("embedding")
        print("vocab size=%d" % self.vocab_size)
        print("embedding dimension=%d" % self.embedding_dim)
        
    def make_layer(self, name='embedder'):
        return Embedding(self.vocab_size, 
            self.embedding_dim, 
            name=name)


class ConvHyper(object):
    def __init__(self, 
            filters, 
            kernel_size=3, 
            strides=2):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
    @property
    def default_name(self):
        return 'cnn'

    @property
    def upsample(self):
        return 1

    @property
    def downsample(self):
        return self.strides

    @property
    def return_sequences(self):
        return True

    @staticmethod
    def random(r):
        filters = r.choice([2 ** i for i in range(6, 10)])
        filters = int(filters)
        kernel_size = r.randint(8) + 2
        strides = r.randint(4) + 1
        return ConvHyper(filters, kernel_size, strides)
        
    def display(self):
        print("conv 1d")
        print("filters=%d" % self.filters)
        print("kernel size=%d" % self.kernel_size)
        print("strides = %d" % self.strides)
        
    def make_layer(self, name='cnn'):
        return Conv1D(self.filters, 
            self.kernel_size, 
            strides=self.strides, 
            padding='causal', 
            activation='relu', 
            name=name)


class RnnHyper(object):
    def __init__(self, 
            hidden_dim, 
            is_lstm, 
            is_bidirectional, 
            return_sequences,
            dropout=0,
            unroll=False):
        self.hidden_dim = hidden_dim
        self.is_lstm = is_lstm
        self.is_bidirectional = is_bidirectional
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.unroll = unroll
        
    @property
    def default_name(self):
        return 'rnn'

    @property
    def upsample(self):
        return 1

    @property
    def downsample(self):
        return 1

    @staticmethod
    def random(r, return_sequences):
        hidden_dim = r.choice([2 ** i for i in range(6, 10)])
        hidden_dim = int(hidden_dim)
        is_lstm = bool(r.randint(2))
        is_bidirectional = bool(r.randint(2))
        return RnnHyper(hidden_dim, 
            is_lstm, 
            is_bidirectional, 
            return_sequences=return_sequences)

    @property
    def output_dim(self):
        if self.is_bidirectional:
            return 2 * self.hidden_dim
        return self.hidden_dim

    def display(self):
        print("RNN")
        print("hidden dimension=%d" % self.hidden_dim)
        if self.is_bidirectional:
            print("bidirectional")
        if self.is_lstm:
            print("lstm")
        else:
            print("gru")
        if self.return_sequences:
            print("return sequences")            
        
    def make_layer(self, name='rnn'):
        if self.is_lstm:
            make_rnn = LSTM
        else:
            make_rnn = GRU
        if self.is_bidirectional:
            rnn = make_rnn(self.hidden_dim, 
                return_sequences=self.return_sequences,
                dropout=self.dropout,
                unroll=self.unroll)
            return Bidirectional(rnn, name=name)
        return make_rnn(self.hidden_dim, 
            return_sequences=self.return_sequences, 
            name=name,
            dropout=self.dropout,
            unroll=self.unroll)

class Deconv(object):
    def __init__(self, upsample, conv):
        self.upsample = upsample
        self.conv = conv

    def __call__(self, x):
        h = self.upsample(x)
        return self.conv(h)


class DeconvHyper(object):
    def __init__(self, 
            filters, 
            kernel_size=3, 
            upsample=2):
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample = upsample
        
    @property
    def default_name(self):
        return 'dcnn'

    @property
    def downsample(self):
        return 1

    @property
    def return_sequences(self):
        return True

    @staticmethod
    def random(r, upsample=None):
        filters = r.choice([2 ** i for i in range(6, 10)])
        filters = int(filters)
        kernel_size = r.randint(8) + 2
        if upsample is None:
            upsample = r.randint(4) + 1
        return DeconvHyper(filters, kernel_size, upsample)
        
    def display(self):
        print("deconv 1d")
        print("filters=%d" % self.filters)
        print("kernel size=%d" % self.kernel_size)
        print("upsample = %d" % self.upsample)
        
    def make_layer(self, name='dcnn'):
        conv = Conv1D(self.filters, 
            self.kernel_size, 
            strides=1, 
            padding='causal', 
            activation='relu', 
            name=name)
        upsample = UpSampling1D(self.upsample)
        return Deconv(upsample, conv)
