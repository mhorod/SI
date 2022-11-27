import random
import math

from text_model import TextModel

class NgramModel(TextModel):
    '''
    n-gram model predicts nth token based on n-1 previous tokens
    '''
    def __init__(self, n, text, tokenize):
        self.n = n
        self.tokens = tokenize(text)
        self.tokenize = tokenize

        self.prefixes = {}
        self.probabilities = {}
        self.top_level_prefixes = []

        self.generate_prefixes()

    def generate_prefixes(self):
        for n in range(1, self.n):
            for i in range(len(self.tokens) - n):
                prefix = tuple(self.tokens[i:i+n])
                next_token = self.tokens[i+n]
                self.prefixes[prefix] = self.prefixes.get(prefix, []) + [next_token]
                if n == self.n - 1:
                    self.top_level_prefixes.append(prefix)

                if prefix in self.probabilities:
                    self.probabilities[prefix][next_token] = self.probabilities[prefix].get(next_token, 0) + 1
                else:
                    self.probabilities[prefix] = {next_token: 1}

        for prefix in self.probabilities:
            total = sum(self.probabilities[prefix].values())
            for token in self.probabilities[prefix]:
                self.probabilities[prefix][token] /= total


    def next(self, prefix):
        if prefix in self.prefixes:
            return random.choice(self.prefixes[prefix])
        else:
            return random.choice(self.tokens)

    def generate_from_prompt(self, prompt, length):
        current = self.tokenize(prompt)
        return self.generate_from_prefix(current, length)

    def generate_from_prefix(self, prefix, length):
        current = list(prefix)
        prefix = current[-(self.n - 1):]

        while len(current) < length:
            next = self.next(tuple(prefix))
            current.append(next)
            prefix = current[-(self.n - 1):]
        return current

    def generate(self, length):
        start = random.choice(self.top_level_prefixes)
        return self.generate_from_prefix(start, length)


    def probability(self, sequence):
        prefix = tuple(sequence[:-1])
        token = sequence[-1]
        for i in range(len(prefix)):
            if prefix[i:] in self.probabilities and token in self.probabilities[prefix[i:]]:
                return self.probabilities[prefix[i:]][token]
        return 1 / len(self.tokens)


    def perplexity(self, text):
        tokens = self.tokenize(text)
        total_log = 0
        for i in range(len(tokens) - self.n):
            sequence = tokens[i:i+self.n]
            total_log += math.log(self.probability(sequence))
        return math.exp(-total_log / len(tokens))

def tokenize_into_chars(text):
    return list(text)

def tokenize_into_words(text):
    new_text = ''
    for c in text:
        if c.isalpha() or c == ' ':
            new_text += c
        else:
            new_text += " "
    words = new_text.split(' ')
    # Remove empty strings
    return [w for w in words if w]


def new_line_after_punctuation(text):
    return text.replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')