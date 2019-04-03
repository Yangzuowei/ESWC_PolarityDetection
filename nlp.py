from nltk.tokenize import word_tokenize
import re


def tokenize(text):
    text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
    return [token.lower() for token in word_tokenize(text)]
