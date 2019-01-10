import unittest

from hyper_params import *
import text_decoder as td


class TestTextDecoder(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_properties(self):
        embed_h = EmbeddingHyper(256, 103)
        conv_h = ConvHyper(89, 6, 4)
        rnn_h = RnnHyper(101, is_lstm=False, is_bidirectional=False, return_sequences=False)
        hyper = te.Hyper(embed_h, [conv_h, rnn_h])

        self.assertEqual(256, hyper.vocab_size)
        self.assertEqual(103, hyper.embedding_dim)
        self.assertFalse(hyper.return_sequences)
        self.assertEqual(101, hyper.encoding.hidden_dim)
        self.assertFalse(hyper.encoding.is_lstm)
        self.assertFalse(hyper.encoding.is_bidirectional)

        self.assertTrue(False, 'test dropout')

    def test_make_layers(self):
        self.assertTrue(False)

    def test_model_shapes(self):
        self.assertTrue(False)

    def test_predict(self):
        self.assertTrue(False)
