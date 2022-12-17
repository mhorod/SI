import numpy as np

from prepare_dataset import standardize_text

def bleu(original, translated, max_ngram=4):

    # Standardize and remove <bos> and <eos> tokens
    original_tokens = standardize_text(original).numpy().decode('utf-8').split()[1:-1]
    translated_tokens = standardize_text(translated).numpy().decode('utf-8').split()[1:-1]

    original_ngrams = get_ngrams(original_tokens, max_ngram)
    translated_ngrams = get_ngrams(translated_tokens, max_ngram)

    n_gram_product = 1

    max_ngram = min(max_ngram, len(original_tokens), len(translated_tokens))
    for n in range(1, max_ngram + 1):
        common = 0
        for ngram in original_ngrams[n]:
            if ngram in translated_ngrams[n]:
                common += 1
        n_gram_product *= (common / len(translated_ngrams[n])) ** (2 ** -n)
            

    length_penalty = min(1, np.exp(1 - len(original_tokens) / len(translated_tokens)))
    return length_penalty * n_gram_product
    
def get_ngrams(tokens, max_ngram):
    '''
    Get the n-grams for a sequence of tokens
    '''
    return {
        n : set([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] )
            for n in range(1, max_ngram + 1)
    }



def bleu_with_preprocessor(original, translated, preprocessor):
    '''
    Calculate the BLEU score for a translated sentence
    Both arguments should be strings representing sentences
    '''
    original_tokens = preprocessor(original)
    translated_tokens = preprocessor(translated)
    return bleu(original_tokens, translated_tokens)