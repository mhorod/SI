import random

import tensorflow as tf
from matplotlib import pyplot as plt

import attention_model
import vanilla_model

from prepare_dataset import *
from bleu import *

def masked_loss(y_true, y_pred):
    # Because we pad tokens with 0, we have to calculate the loss only on the non-zero tokens
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(y_true != 0, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    # Same thing for accuracy
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_true.dtype)
    matched = tf.cast(tf.equal(y_true, y_pred), dtype=y_true.dtype)
    mask = tf.cast(y_true != 0, dtype=matched.dtype)
    return tf.reduce_sum(matched) / tf.reduce_sum(mask)

def train_and_save_model(name, model, train_ds, epochs=20):
    model.compile(optimizer='adam', loss=masked_loss,
                metrics=[masked_accuracy, masked_loss])
    history = model.fit(train_ds, epochs=epochs)
    tf.saved_model.save(model, f"models/{name}")

    fig, ax = plt.subplots(1, 1)
    ax.plot(history.history['masked_accuracy'], label='Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(f"{name}-history.png")


def evaluate_model(model, ds):
    _, acc, loss = model.evaluate(ds)
    print(f"Accuracy: {acc}, Loss: {loss}")


def get_bleu_scores(model, sequences):
    scores = []
    texts = [s[0] for s in sequences]
    targets = [s[1] for s in sequences]
    translations = model.translate(texts)
    for text, target, prediction in zip(texts, targets, translations):
        bleu_score = bleu(target, prediction)
        print(f"{text} ({target}) -> {prediction} (BLEU: {bleu_score:.2f})")
        scores.append(bleu_score)
    avg = sum(scores) / len(scores)
    print(f"Average BLEU score: {avg:.2f}")

def bleu_scores_train_test(model, train_sample, test_sample):
    print("Train set")
    get_bleu_scores(model, train_sample)
    print()
    print("Test set")
    get_bleu_scores(model, test_sample)

MAX_VOCAB_SIZE = 10**6
train_coeff = 0.8

translations = load_translations_from_file("spa-big.txt")
translations, test_translations = translations[int(len(translations) * train_coeff):], translations[:int(len(translations) * train_coeff)]

train_ds, test_ds = datasets_from_translations(translations)

context_text_preprocessor, target_text_preprocessor = make_text_preprocessors(
    MAX_VOCAB_SIZE, train_ds)
train_ds = process_dataset(
    train_ds, context_text_preprocessor, target_text_preprocessor)

test_ds = process_dataset(
    test_ds, context_text_preprocessor, target_text_preprocessor)

RNN_UNITS = 1024
vanilla = vanilla_model.Seq2Seq(context_text_preprocessor, target_text_preprocessor, RNN_UNITS)
attention = attention_model.Seq2Seq(context_text_preprocessor, target_text_preprocessor, RNN_UNITS)

# Choose 10 random sentences from train translations
train_sample = random.sample(translations, 100)
test_sample = random.sample(test_translations, 100)

train_and_save_model("vanilla", vanilla, train_ds, epochs=30)
evaluate_model(vanilla, test_ds)
bleu_scores_train_test(vanilla, train_sample, test_sample)

train_and_save_model("attention", attention, train_ds, epochs=30)
evaluate_model(attention, test_ds)
bleu_scores_train_test(attention, train_sample, test_sample)