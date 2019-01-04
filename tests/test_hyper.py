import unittest
import numpy as np

import hyper_params as hp

class TestHyper(unittest.TestCase):
    def setUp(self):
        self.r = np.random.RandomState(42)

    def tearDown(self):
        pass

    def testHyper(self):
        hyper = hp.Hyper()
        self.assertEqual(0.001, hyper.lr)
        self.assertEqual(10, hyper.batch_size)
        self.assertEqual(3, hyper.epochs)

        hyper = hp.Hyper(lr=0.1, batch_size=5, epochs=4)
        self.assertEqual(0.1, hyper.lr)
        self.assertEqual(5, hyper.batch_size)
        self.assertEqual(4, hyper.epochs)

        hyper = hp.Hyper.Random(self.r)
        self.assertEqual(0.001 * 0.1 ** 0.5, hyper.lr)
        self.assertEqual(5, hyper.batch_size)
        self.assertEqual(3, hyper.epochs)

        actual = hyper.make_optimizer()
        self.assertIsNotNone(actual)

    def testEmbedder(self):
        embedder = hp.EmbeddingHyper()
        self.assertEqual(256, embedder.vocab_size)
        self.assertEqual(64, embedder.embedding_dim)

        embedder = hp.EmbeddingHyper(vocab_size=1024, embedding_dim=128)
        self.assertEqual(1024, embedder.vocab_size)
        self.assertEqual(128, embedder.embedding_dim)

        embedder = hp.EmbeddingHyper.Random(self.r)
        self.assertEqual(256, embedder.vocab_size)
        self.assertEqual(256, embedder.embedding_dim)

        actual = embedder.make_layer()
        self.assertEqual('embedder', actual.name)

    def testConv(self):
        conv = hp.ConvHyper(128)
        self.assertEqual(128, conv.filters)
        self.assertEqual(3, conv.kernel_size)
        self.assertEqual(2, conv.strides)

        conv = hp.ConvHyper(512, kernel_size=6, strides=4)
        self.assertEqual(512, conv.filters)
        self.assertEqual(6, conv.kernel_size)
        self.assertEqual(4, conv.strides)

        conv = hp.ConvHyper.Random(self.r)
        self.assertEqual(256, conv.filters)
        self.assertEqual(5, conv.kernel_size)
        self.assertEqual(1, conv.strides)

        actual = conv.make_layer()
        self.assertEqual('cnn', actual.name)

    def testDeconv(self):
        dconv = hp.DeconvHyper(128)
        self.assertEqual(128, dconv.filters)
        self.assertEqual(3, dconv.kernel_size)
        self.assertEqual(2, dconv.upsample)

        dconv = hp.DeconvHyper(512, kernel_size=6, upsample=4)
        self.assertEqual(512, dconv.filters)
        self.assertEqual(6, dconv.kernel_size)
        self.assertEqual(4, dconv.upsample)

        dconv = hp.DeconvHyper.Random(self.r)
        self.assertEqual(256, dconv.filters)
        self.assertEqual(5, dconv.kernel_size)
        self.assertEqual(1, dconv.upsample)

        actual, upsample = dconv.make_layers()
        self.assertEqual('dcnn', actual.name)
        self.assertIsNotNone(upsample)

    def testRnn(self):
        rnn = hp.RnnHyper(128,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=False)
        self.assertEqual(128, rnn.hidden_dim)
        self.assertFalse(rnn.is_lstm)
        self.assertTrue(rnn.is_bidirectional)
        self.assertFalse(rnn.return_sequences)

        rnn = hp.RnnHyper(512,
            is_lstm=False,
            is_bidirectional=True,
            return_sequences=True)
        self.assertEqual(512, rnn.hidden_dim)
        self.assertFalse(rnn.is_lstm)
        self.assertTrue(rnn.is_bidirectional)
        self.assertTrue(rnn.return_sequences)

        rnn = hp.RnnHyper.Random(self.r, return_sequences=False)
        self.assertEqual(256, rnn.hidden_dim)
        self.assertTrue(rnn.is_lstm)
        self.assertFalse(rnn.is_bidirectional)
        self.assertFalse(rnn.return_sequences)

        actual = rnn.make_layer()
        self.assertEqual('rnn', actual.name)
