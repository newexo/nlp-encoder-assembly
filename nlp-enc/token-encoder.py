"""
This is an initial attempt at describing an encoder.
Most of this code is taken from 

https://raw.githubusercontent.com/fchollet/keras/master/examples/lstm_seq2seq.py

see /reading-club/eaten_by_a_grue/one_hot.py 11/1/18
"""

import numpy as np


def make_token_dict(data):
    """
    :param data: this should be text loaded into an array
    we don't care if strings are of equal length
    :returns: a dictionary whose keys are tokens, the values are the indices
    """

    input_characters = set()

    for input_text in data:
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)


    input_characters = sorted(list(input_characters))
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])

    return input_token_index


def compute_encoder_parameters(data):
    """
    this will get the max length of text strings in the data

    :return: int: a length of the longest string
    """
    return max([len(txt) for txt in data])


def one_hot(input_token_index, data):
    """
    This will create an encoding from tokenized data
    :param input_token_index: dictionary of tokens
    :param data: text data as list of lists
    :return: an encoding as a list of lists
    """

    encoder_input_data = np.zeros(
        (len(data), compute_encoder_parameters(data), len(input_token_index)), dtype='float32')
    for i, input_text,  in enumerate(data):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.

    return encoder_input_data
