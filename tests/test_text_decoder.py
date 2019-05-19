import unittest

from hyper_params import *
import text_decoder as td


class TestTextDecoder(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_make_layers(self):
        rnn_h = RnnHyper(256, 
            is_lstm=False,
            is_bidirectional=False,
            return_sequences=True, 
            unroll=True)
        deconv_h = DeconvHyper(128)
        hyper = td.Hyper(256, [rnn_h, deconv_h])

        decoder = hyper.make_layer()
        rnn, deconv = decoder.layers
        self.assertEqual('decoder_rnn', rnn.name)
        self.assertEqual('decoder_dcnn', deconv.conv.name)
        self.assertEqual('probs', decoder.dense.name)

    def test_model_shapes(self):
        rnn_h = RnnHyper(256, 
            is_lstm=False,
            is_bidirectional=False,
            return_sequences=True, 
            unroll=True)
        deconv_h = DeconvHyper(128, 6, 4)
        hyper = td.Hyper(256, [rnn_h, deconv_h])

        decoder = hyper.make_layer()
        x = Input(shape=(128,), name='text_input')

        h = decoder(x, 64)
        self.assertEqual(3, len(h.shape))
        self.assertEqual(64, h.shape[1])
        self.assertEqual(256, h.shape[2])

        h = decoder(x, 128)
        self.assertEqual(3, len(h.shape))
        self.assertEqual(128, h.shape[1])
        self.assertEqual(256, h.shape[2])
