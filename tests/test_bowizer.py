import unittest
import numpy as np
from numpy.linalg import norm

import bowizer


class TestBowizer(unittest.TestCase):
    def setUp(self):
        self.three_lines = """Midway upon the journey of our life
I found myself within a forest dark,
For the straightforward pathway had been lost."""
        self.lines = self.three_lines.split('\n')
        self.text = """Midway upon the journey of our life
I found myself within a forest dark,
For the straightforward pathway had been lost.

Ah me! how hard a thing it is to say
What was this forest savage, rough, and stern,
Which in the very thought renews the fear.

So bitter is it, death is little more;
But of the good to treat, which there I found,
Speak will I of the other things I saw there.

I cannot well repeat how there I entered,
So full was I of slumber at the moment
In which I had abandoned the true way.

But after I had reached a mountain's foot,
At that point where the valley terminated,
Which had with consternation pierced my heart,

Upward I looked, and I beheld its shoulders,
Vested already with that planet's rays
Which leadeth others right by every road.

Then was the fear a little quieted
That in my heart's lake had endured throughout
The night, which I had passed so piteously.

And even as he, who, with distressful breath,
Forth issued from the sea upon the shore,
Turns to the water perilous and gazes;

So did my soul, that still was fleeing onward,
Turn itself back to re-behold the pass
Which never yet a living person left.

After my weary body I had rested,
The way resumed I on the desert slope,
So that the firm foot ever was the lower."""

    def tearDown(self):
        pass

    def test_word_tokenizer(self):
        actual = bowizer.word_tokenize(self.three_lines)
        expected = ['Midway', 'upon', 'the', 'journey', 'of', 'our', 'life', 'I', 'found', 'myself', 'within', 'a', 'forest', 'dark', ',', 'For', 'the', 'straightforward', 'pathway', 'had', 'been', 'lost', '.']
        self.assertEqual(expected, actual)

    def test_get_vocab(self):
        tokenized_text = bowizer.word_tokenize(self.text)
        vocab_size = 5
        extendVocabList, td, vocab = bowizer.get_vocab(tokenized_text, vocab_size)

        expected = ['unk', ',', 'the', 'I', '.', 'had']
        actual = extendVocabList
        self.assertEqual(expected, actual)

        expected = {'unk': 0, ',': 1, 'the': 2, 'I': 3, '.': 4, 'had': 5}
        actual = td
        self.assertEqual(expected, actual)

        expected = {',', 'the', 'I', '.', 'had'}
        actual = vocab
        self.assertEqual(expected, actual)

    def test_tokenize_docs(self):
        actual = bowizer.tokenize_docs(self.lines)
        expected = ['Midway', 'upon', 'the', 'journey', 'of', 'our', 'life', 'I', 'found', 'myself', 'within', 'a', 'forest', 'dark', ',', 'For', 'the', 'straightforward', 'pathway', 'had', 'been', 'lost', '.']
        self.assertEqual(expected, actual)

    def test_BOW(self):
        tokenized_text = bowizer.word_tokenize(self.text)
        vocab_size = 5
        extendVocabList, td, vocab = bowizer.get_vocab(tokenized_text, vocab_size)
        wizer = bowizer.BOWizer(extendVocabList, td, vocab)
        actual = wizer.bow(self.text)
        expected = np.array([0.75091575, 0.08058608, 0.06227106, 0.05128205, 0.02930403, 0.02564103])
        self.assertAlmostEqual(0, norm(expected - actual))
