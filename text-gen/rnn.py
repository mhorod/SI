import tensorflow as tf

from prepare_dataset import *
from rnn_model import *


with open("pan-tadeusz.txt", "r") as f:
    text = f.read()

symbols = '.,:;?!'

for symbol in symbols:
    text = text.replace(symbol, ' ')

text = text.split(' ')

text = [word for word in text if word]

train_text = text[:int(len(text)*0.8)]
test_text = text[int(len(text)*0.8):]


train_dataset, lookup = dataset_from_text(text)
test_dataset = dataset_with_lookup(test_text, lookup)


model = BaseModel(len(lookup.ids_from_chars.get_vocabulary()), embedding_dim=1024, rnn_units=2048)

vocab_size = len(lookup.ids_from_chars.get_vocabulary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(train_dataset, epochs=40)

train_perplexity = tf.exp(model.evaluate(train_dataset))
test_perplexity = tf.exp(model.evaluate(test_dataset))


print("-----")
print(f"Train perplexity: {train_perplexity}")
print(f"Test perplexity: {test_perplexity}")
print("-----")

# Save model
model.save('model')

one_step_model = TextGenModel(model, lookup)

rnn_model = RnnModel(one_step_model, model)

print("Samle RNN text: ", rnn_model.generate(100))

with open("rnn-examples.txt", "w") as f:
    for _ in range(10):
        text = rnn_model.generate(100)
        f.write(text)
        f.write("\n\n")

tf.saved_model.save(one_step_model, "one_step_model")
