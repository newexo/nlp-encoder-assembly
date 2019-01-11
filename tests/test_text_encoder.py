import unittest

from hyper_params import *
import text_encoder as te


class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_properties(self):
        embed_h = EmbeddingHyper(256, 103)
        conv_h = ConvHyper(89, 6, 4)
        rnn_h = RnnHyper(101,
                         is_lstm=False,
                         is_bidirectional=False,
                         return_sequences=True,
                         dropout=0.1)
        hyper = te.Hyper(embed_h, [conv_h, rnn_h])

        self.assertEqual(256, hyper.vocab_size)
        self.assertEqual(103, hyper.embedding_dim)
        self.assertTrue(hyper.return_sequences)
        self.assertEqual(101, hyper.encoding.hidden_dim)
        self.assertFalse(hyper.encoding.is_lstm)
        self.assertFalse(hyper.encoding.is_bidirectional)
        self.assertEqual(0.1, hyper.encoding.dropout)

        self.assertEqual(1, hyper.upsample)
        self.assertEqual(4, hyper.downsample)

    def test_make_layers(self):
        embed_h = EmbeddingHyper(256, 103)
        conv_h = ConvHyper(89, 6, 4)
        rnn_h = RnnHyper(101,
                         is_lstm=False,
                         is_bidirectional=False,
                         return_sequences=False,
                         dropout=0.1)
        hyper = te.Hyper(embed_h, [conv_h, rnn_h])
        encoder = hyper.make_layer()

        self.assertEqual('embedder', encoder.embedder.name)

        conv, rnn = encoder.layers
        self.assertEqual('encoder_cnn', conv.name)
        self.assertEqual('encoder_rnn', rnn.name)

    def test_model_shapes(self):
        embed_h = EmbeddingHyper(256, 103)
        conv_h = ConvHyper(89, 6, 4)
        rnn_h = RnnHyper(101,
                         is_lstm=False,
                         is_bidirectional=False,
                         return_sequences=False,
                         dropout=0.1)
        hyper = te.Hyper(embed_h, [conv_h, rnn_h])
        encoder = hyper.make_layer()
        x = Input(shape=(128,), name='text_input')
        h = encoder(x)

        self.assertEqual(2, len(h.shape))
        self.assertEqual(101, int(h.shape[1]))
