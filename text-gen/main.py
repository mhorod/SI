from ngram_model import *
from rnn_model import *


TEXT_PATH = 'pan-tadeusz.txt'
text = open(TEXT_PATH).read()

train_to_test_ratio = 0.8

train_text = text[:int(len(text) * train_to_test_ratio)]
test_text = text[int(len(text) * train_to_test_ratio):]


def measure_perplexity(model_name, model, test_text):
    print(f"Perplexity for {model_name}: {model.perplexity(test_text)}")


def generate_text(model_name, model, length):
    print(f"Generated text for {model_name}: {model.generate(length)}")


def main():
    ngram_model = NgramModel(3, train_text, tokenize_into_words)
    rnn_model = RnnModel.from_file("model")

    print("TRAIN PERPLEXITY")
    measure_perplexity("ngram", ngram_model, train_text)
    measure_perplexity("rnn", rnn_model, train_text)

    print()

    print("TEST PERPLEXITY")
    measure_perplexity("ngram", ngram_model, test_text)
    measure_perplexity("rnn", rnn_model, test_text)

    print()

    generate_text("ngram", ngram_model, 100)
    generate_text("rnn", rnn_model, 1000)


main()
