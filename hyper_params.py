from keras.layers import Embedding, Conv1D, Input, GRU, LSTM, Bidirectional, Dense, UpSampling1D, Dropout, TimeDistributed, RepeatVector
from keras.optimizers import Adam


def name_prefix(prefix):
    if prefix is None or not len(prefix.strip()):
        return ''
    return '%s' % prefix

def rename(prefix, name):
    prefix = name_prefix(prefix)
    return prefix + name

class Hyper(object):
    def __init__(self, 
            lr=0.001, 
            batch_size=10, 
            epochs=3):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def Random(r):
        lr = r.choice([0.01 * (10 ** -(0.5 * i)) for i in range(6)])
        batch_size = r.choice([5, 10, 20, 50])
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
        
    @staticmethod
    def Random(r):
        embedding_dim = r.choice([2 ** i for i in range(6, 10)])
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
        
    @staticmethod
    def Random(r):
        filters = r.choice([2 ** i for i in range(6, 10)])
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
            return_sequences):
        self.hidden_dim = hidden_dim
        self.is_lstm = is_lstm
        self.is_bidirectional = is_bidirectional
        self.return_sequences = return_sequences
        
    @staticmethod
    def Random(r, return_sequences):
        hidden_dim = r.choice([2 ** i for i in range(6, 10)])
        is_lstm = bool(r.randint(2))
        is_bidirectional = bool(r.randint(2))
        return RnnHyper(hidden_dim, 
            is_lstm, 
            is_bidirectional, 
            return_sequences=return_sequences)

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
                return_sequences=self.return_sequences)
            return Bidirectional(rnn, name=name)
        return make_rnn(self.hidden_dim, 
            return_sequences=self.return_sequences, 
            name=name)
    
class DeconvHyper(object):
    def __init__(self, 
            filters, 
            kernel_size=3, 
            upsample=2):
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample = upsample
        
    @staticmethod
    def Random(r, upsample=None):
        filters = r.choice([2 ** i for i in range(6, 10)])
        kernel_size = r.randint(8) + 2
        if upsample is None:
            upsample = r.randint(4) + 1
        return DeconvHyper(filters, kernel_size, upsample)
        
    def display(self):
        print("deconv 1d")
        print("filters=%d" % self.filters)
        print("kernel size=%d" % self.kernel_size)
        print("upsample = %d" % self.upsample)
        
    def make_layers(self, name='dcnn'):
        conv = Conv1D(self.filters, 
            self.kernel_size, 
            strides=1, 
            padding='causal', 
            activation='relu', 
            name=name)
        return conv, UpSampling1D(self.upsample)

class RnnCnnHyper(object):
    def __init__(self, embedder, conv, rnn):
        self.embedder = embedder
        self.conv = conv
        self.rnn = rnn
        
    @staticmethod
    def Random(r):
        embedder = EmbeddingHyper.Random(r)
        conv = ConvHyper.Random(r)
        rnn = RnnHyper.Random(r)
        
        return RnnCnnHyper(embedder, conv, rnn)

    def display(self):
        self.embedder.display()
        print()
        self.conv.display()
        print()
        self.rnn.display()
        print()
        
    def make_layers(self, name, return_sequences):
        if name is not None and len(name):
            prefix = '%s_' % name
        else:
            prefix = ''
        embedder = self.embedder.make_layer(name='%sembedder' % prefix)
        conv = self.conv.make_layer(name='%sconv' % prefix)
        rnn = self.rnn.make_layer(name='%srnn' % prefix, return_sequences=return_sequences)
        dense = Dense(self.embedder.vocab_size, activation='softmax', name='%sprobs' % prefix)
        return embedder, conv, rnn, dense
    
