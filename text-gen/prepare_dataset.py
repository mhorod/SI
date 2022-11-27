'''
Prepare dataset for tensorflow RNN
'''

import tensorflow as tf
from dataclasses import dataclass

@dataclass
class Lookup:
    '''
    Translation between characters and tensor ids.
    '''
    ids_from_chars: tf.keras.layers.StringLookup
    chars_from_ids: tf.keras.layers.StringLookup


def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def split_sequence_into_input_and_output(sequence):
    input = sequence[:-1]
    output = sequence[1:]
    return input, output

def lookup_from_text(text):
    # Extract all characters from the text
    chars = sorted(list(set(text)))

    # Create a mapping from characters to numbers
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=chars, 
        mask_token=None)

    # Invert the mapping
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), 
        mask_token=None, 
        invert=True)
    
    return Lookup(ids_from_chars, chars_from_ids)

def dataset_from_text(text, batch_size=64, sequence_length=128):
    # Generate lookup
    lookup = lookup_from_text(text)
    dataset = dataset_with_lookup(text, lookup, batch_size, sequence_length)
    return dataset, lookup


def dataset_with_lookup(text, lookup, batch_size=64, sequence_length=128):
    # Convert text to ids using the lookup
    ids = lookup.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    # Create a dataset of character ids
    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)

    # Split into sequences
    sequences = ids_dataset.batch(sequence_length, drop_remainder=True)

    # Split into input and expected output
    dataset = sequences.map(split_sequence_into_input_and_output)

    # Shuffle and batch
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    return dataset

def dataset_from_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return dataset_from_text(text)