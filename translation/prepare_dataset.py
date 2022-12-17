import tensorflow as tf
import numpy as np

def load_translations_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    translations = [
        tuple(line.split('\t')[:2]) for line in lines
    ]

    return translations


def datasets_from_translations(translations, test_size=0.2):
    context = np.array([t[0] for t in translations])
    target = np.array([t[1] for t in translations])

    is_train = np.random.rand(len(translations)) < (1 - test_size)

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((context[is_train], target[is_train]))
        .shuffle(1000)
        .batch(64)
    )

    test_raw = (
        tf.data.Dataset
        .from_tensor_slices((context[~is_train], target[~is_train]))
        .shuffle(1000)
        .batch(64)

    )

    return train_raw, test_raw

def standardize_text(text):
    # lowercase
    text = tf.strings.lower(text)
    # add space around punctuation
    text = tf.strings.regex_replace(text, r"([?¡.,¿!])", r" \1 ")
    # remove extra spaces
    text = tf.strings.regex_replace(text, r'[" "]+', " ")

    # add start and end tokens
    text = tf.strings.join(["<bos>", text, "<eos>"], separator=" ")
    return text


def make_text_preprocessors(max_vocab_size, ds):
    context_text_preprocessor = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size,
        standardize=standardize_text,
        ragged=True,
    )
    context_text_preprocessor.adapt(ds.map(lambda x, y: x))

    target_text_preprocessor = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size,
        standardize=standardize_text,
        ragged=True,
    )
    target_text_preprocessor.adapt(ds.map(lambda x, y: y))

    return context_text_preprocessor, target_text_preprocessor


def process_dataset(ds, context_preprocessor, target_preprocessor):
    def process_text(context, target):
        context = context_preprocessor(context).to_tensor()
        target = target_preprocessor(target)
        target_input = target[:, :-1].to_tensor()
        target_output = target[:, 1:].to_tensor()
        return (context, target_input), target_output

    return ds.map(process_text)