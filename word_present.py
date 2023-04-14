from functools import lru_cache
from nltk.stem.wordnet import WordNetLemmatizer

wnl = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)

def binary_search(word, word_list):
    left = 0
    right = len(word_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if word_list[mid] == word:
            return True
        elif word_list[mid] < word:
            left = mid + 1
        else:
            right = mid - 1

    return False


def word_present(sentence, flagged_words):
    # Iterate through every word in test sentence
    for word in sentence.split():
        if not binary_search(word, ['a', 'above', 'an', 'and', 'are', 'at', 'being', 'below', 'doing', 'from', 'has', 'in', 'is',
                        'of', 'on', 'the', 'to', 'up']):
            # Lemmatize word
            word = lemmatize(word, 'v')
            # Check if word is in word list
            if binary_search(word, flagged_words):
                return True