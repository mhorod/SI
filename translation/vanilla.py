import tensorflow as tf

import random

from prepare_dataset import *
from bleu import *

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, rnn_units):
        super().__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(
                rnn_units, return_sequences=True)
        )

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, rnn_units, mask_zero=True)

    def call(self, x, training=False):
        x = self.embedding(x)
        return self.rnn(x, training=training)

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        return self(self.text_processor(texts).to_tensor())


class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, rnn_units):
        super().__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()

        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(), mask_token='')

        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(), mask_token='', invert=True)

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, rnn_units, mask_zero=True)

        self.attention = tf.keras.layers.Attention()

        self.rnn = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True)

        self.dense = tf.keras.layers.Dense(self.vocab_size)

        self.start_token = self.word_to_id('<bos>')
        self.end_token = self.word_to_id('<eos>')

    def call(self, x, state=None, training=False, return_state=False):
        x, ctx = x
        x = self.embedding(x)

        
        ctx = ctx[:, -1, :]

        ctx = tf.tile(tf.expand_dims(ctx, 1), [1, tf.shape(x)[1], 1])

        x = tf.concat([x, ctx], axis=-1)
        
        x, state = self.rnn(x, initial_state=state, training=training)

        x = self.dense(x)
        if return_state:
            return x, state
        else:
            return x

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(tokens)
        
        ctx = context[:, -1, :]
        ctx = tf.tile(tf.expand_dims(ctx, 1), [1, tf.shape(embedded)[1], 1])
        embedded = tf.concat([embedded, ctx], axis=-1)

        state = self.rnn.get_initial_state(embedded)[0]
        return tokens, state, done

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy()

    def get_next_token(self, next_token, context, state, done):
        logits, state = self((next_token, context), state, return_state=True)
        next_token = tf.argmax(logits, axis=-1)
        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, state, done


class Seq2Seq(tf.keras.Model):
    def __init__(self, context_text_processor, target_text_processor, rnn_units):
        super().__init__()
        self.encoder = Encoder(context_text_processor, rnn_units)
        self.decoder = Decoder(target_text_processor, rnn_units)

    def call(self, inputs):
        context, target = inputs
        encoder_states = self.encoder(context)
        decoded = self.decoder((target, encoder_states))
        return decoded

    def translate(self, texts):
        context = self.encoder.convert_input(texts)
        tokens = []
        next_token, state, done = self.decoder.get_initial_state(context)

        for _ in range(10):
            next_token, state, done = self.decoder.get_next_token(
                next_token, context, state, done)
            tokens.append(next_token)

        tokens = tf.concat(tokens, axis=1)
        result = self.decoder.tokens_to_text(tokens)
        return result


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


MAX_VOCAB_SIZE = 10**6

train_coeff = 0.8

translations = load_translations_from_file("spa-small.txt")
translations, test_translations = translations[int(len(translations) * train_coeff):], translations[:int(len(translations) * train_coeff)]


train_ds, test_ds = datasets_from_translations(translations)

context_text_preprocessor, target_text_preprocessor = make_text_preprocessors(
    MAX_VOCAB_SIZE, train_ds)
train_ds = process_dataset(
    train_ds, context_text_preprocessor, target_text_preprocessor)

test_ds = process_dataset(
    test_ds, context_text_preprocessor, target_text_preprocessor)

RNN_UNITS = 1024
model = Seq2Seq(context_text_preprocessor, target_text_preprocessor, RNN_UNITS)

model.compile(optimizer='adam', loss=masked_loss,
              metrics=[masked_accuracy, masked_loss])
model.fit(train_ds, epochs=50)

# evaluate on test set
loss, acc, _ =  model.evaluate(test_ds)
print(f"EVALUTATION - Loss: {loss}, Accuracy: {acc}")


# Choose 10 random sentences from train translations
train_sample = random.sample(translations, 10)
test_sample = random.sample(test_translations, 10)

# Translate them

print("Train sample:")
for text, target in train_sample:
    translated = model.translate([text])[0].decode('utf-8')
    bleu_score = bleu(target, translated)

    print(f"{text} ({target}) -> {translated} (BLEU: {bleu_score:.2f})")


print("Test sample:")
for text, target in test_sample:
    translated = model.translate([text])[0].decode('utf-8')
    bleu_score = bleu(target, translated)
    print(f"{text} ({target}) -> {translated} (BLEU: {bleu_score:.2f})")