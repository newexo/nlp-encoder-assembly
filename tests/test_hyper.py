import unittest
import numpy as np
from keras.layers import Input

import hyper_params as hp


class TestHyper(unittest.TestCase):
    def setUp(self):
        self.r = np.random.RandomState(42)

    def tearDown(self):
        pass

    def test_hyper(self):
        hyper = hp.Hyper()
        self.assertEqual(0.001, hyper.lr)
        self.assertEqual(10, hyper.batch_size)
        self.assertEqual(3, hyper.epochs)

        hyper = hp.Hyper(lr=0.1, batch_size=5, epochs=4)
        self.assertEqual(0.1, hyper.lr)
        self.assertEqual(5, hyper.batch_size)
        self.assertEqual(4, hyper.epochs)

        hyper = hp.Hyper.random(self.r)
        self.assertEqual(0.001 * 0.1 ** 0.5, hyper.lr)
        self.assertEqual(5, hyper.batch_size)
        self.assertEqual(3, hyper.epochs)

        actual = hyper.make_optimizer()
        self.assertIsNotNone(actual)

    def test_embedder(self):
        embedder = hp.EmbeddingHyper()
        self.assertEqual(256, embedder.vocab_size)
        self.assertEqual(64, embedder.embedding_dim)

        embedder = hp.EmbeddingHyper(vocab_size=1024, embedding_dim=128)
        self.assertEqual(1024, embedder.vocab_size)
        self.assertEqual(128, embedder.embedding_dim)

        embedder = hp.EmbeddingHyper.random(self.r)
        self.assertEqual(256, embedder.vocab_size)
        self.assertEqual(256, embedder.embedding_dim)

        actual = embedder.make_layer()
        self.assertEqual('embedder', actual.name)

    def test_embedder_shapes(self):
        hyper = hp.EmbeddingHyper()
        embedder = hyper.make_layer()
        x = Input(shape=(137,), name='text_input')
        h = embedder(x)
        self.assertEqual(137, int(h.shape[1]))
        self.assertEqual(64, int(h.shape[2]))

    def test_conv(self):
        conv = hp.ConvHyper(128)
        self.assertEqual(128, conv.filters)
        self.assertEqual(3, conv.kernel_size)
        self.assertEqual(2, conv.strides)
        self.assertEqual(1, conv.upsample)
        self.assertEqual(2, conv.downsample)
        self.assertTrue(conv.return_sequences)

        conv = hp.ConvHyper(512, kernel_size=6, strides=4)
        self.assertEqual(512, conv.filters)
        self.assertEqual(6, conv.kernel_size)
        self.assertEqual(4, conv.strides)
        self.assertEqual(1, conv.upsample)
        self.assertEqual(4, conv.downsample)

        conv = hp.ConvHyper.random(self.r)
        self.assertEqual(256, conv.filters)
        self.assertEqual(5, conv.kernel_size)
        self.assertEqual(1, conv.strides)

        actual = conv.make_layer()
        self.assertEqual('cnn', actual.name)

    def test_conv_shapes(self):
        embedder_h = hp.EmbeddingHyper()
        conv_h = hp.ConvHyper(512, kernel_size=6, strides=4)

        embedder = embedder_h.make_layer()
        conv = conv_h.make_layer()

        x = Input(shape=(128,), name='text_input')
        h = embedder(x)
        h = conv(h)

        self.assertEqual(32, int(h.shape[1]))
        self.assertEqual(512, int(h.shape[2]))

    def test_deconv(self):
        dconv = hp.DeconvHyper(128)
        self.assertEqual(128, dconv.filters)
        self.assertEqual(3, dconv.kernel_size)
        self.assertEqual(2, dconv.upsample)
        self.assertEqual(1, dconv.downsample)
        self.assertTrue(dconv.return_sequences)

        dconv = hp.DeconvHyper(512, kernel_size=6, upsample=4)
        self.assertEqual(512, dconv.filters)
        self.assertEqual(6, dconv.kernel_size)
        self.assertEqual(4, dconv.upsample)
        self.assertEqual(1, dconv.downsample)

        dconv = hp.DeconvHyper.random(self.r)
        self.assertEqual(256, dconv.filters)
        self.assertEqual(5, dconv.kernel_size)
        self.assertEqual(1, dconv.upsample)

        actual = dconv.make_layer()
        self.assertEqual('dcnn', actual.conv.name)
        self.assertIsNotNone(actual.upsample)

    def test_deconv_shapes(self):
        hyper = hp.DeconvHyper(64, kernel_size=6, upsample=4)
        embedder_h = hp.EmbeddingHyper()
        embedder = embedder_h.make_layer()
        dec = hyper.make_layer()
        x = Input(shape=(128,), name='text_input')
        h = embedder(x)
        h = dec(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(512, int(h.shape[1]))
        self.assertEqual(64, int(h.shape[2]))

    def test_rnn(self):
        rnn = hp.RnnHyper(128,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=False)
        self.assertEqual(128, rnn.hidden_dim)
        self.assertFalse(rnn.is_lstm)
        self.assertTrue(rnn.is_bidirectional)
        self.assertFalse(rnn.return_sequences)
        self.assertEqual(0, rnn.dropout)
        self.assertEqual(1, rnn.upsample)
        self.assertEqual(1, rnn.downsample)
        self.assertFalse(rnn.unroll)

        rnn = hp.RnnHyper(512,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=True)
        self.assertEqual(512, rnn.hidden_dim)
        self.assertFalse(rnn.is_lstm)
        self.assertTrue(rnn.is_bidirectional)
        self.assertTrue(rnn.return_sequences)

        # can we create gru?
        actual = rnn.make_layer()
        self.assertEqual('rnn', actual.name)

        rnn = hp.RnnHyper.random(self.r, return_sequences=False)
        self.assertEqual(256, rnn.hidden_dim)
        self.assertTrue(rnn.is_lstm)
        self.assertFalse(rnn.is_bidirectional)
        self.assertFalse(rnn.return_sequences)

        # can we create lstm?
        actual = rnn.make_layer()
        self.assertEqual('rnn', actual.name)

        rnn = hp.RnnHyper(128,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=False,
            dropout=0.1)
        self.assertEqual(0.1, rnn.dropout)

        rnn = hp.RnnHyper(128,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=False,
            dropout=0.1,
            unroll=True)
        self.assertTrue(rnn.unroll)

    def test_rnn_shapes(self):
        embedder_h = hp.EmbeddingHyper()
        embedder = embedder_h.make_layer()
        x = Input(shape=(128,), name='text_input')

        rnn_h = hp.RnnHyper(128,
                            is_lstm=False,
                            is_bidirectional=False,
                            return_sequences=False)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(2, len(h.shape))
        self.assertEqual(128, int(h.shape[1]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=False,
                            is_bidirectional=True,
                            return_sequences=False)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(2, len(h.shape))
        self.assertEqual(256, int(h.shape[1]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=False,
                            is_bidirectional=False,
                            return_sequences=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(128, int(h.shape[2]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=False,
                            is_bidirectional=True,
                            return_sequences=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(256, int(h.shape[2]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=False,
                            return_sequences=False)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(2, len(h.shape))
        self.assertEqual(128, int(h.shape[1]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=True,
                            return_sequences=False)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(2, len(h.shape))
        self.assertEqual(256, int(h.shape[1]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=False,
                            return_sequences=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(128, int(h.shape[2]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=True,
                            return_sequences=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(256, int(h.shape[2]))


        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=False,
                            return_sequences=True,
                            unroll=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(128, int(h.shape[1]))
        self.assertEqual(128, int(h.shape[2]))

        rnn_h = hp.RnnHyper(128,
                            is_lstm=True,
                            is_bidirectional=True,
                            return_sequences=True,
                            unroll=True)
        rnn = rnn_h.make_layer()

        h = embedder(x)
        h = rnn(h)

        self.assertEqual(3, len(h.shape))
        self.assertEqual(128, int(h.shape[1]))
        self.assertEqual(256, int(h.shape[2]))
