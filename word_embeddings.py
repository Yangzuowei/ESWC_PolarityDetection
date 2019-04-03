from __future__ import print_function

import numpy as np
from gensim.models import KeyedVectors
import os


def load(path, vocabulary=None):
    print('Loading word embeddings at', path)
    embeddings = {}
    counter = 0
    with open(path) as f:
        for line in f:
            counter += 1
            try:
                values = line.split()
                word = values[0]
                if (vocabulary is None) or (word in vocabulary):
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings[word] = vector
            except IndexError:
                print('Index error at line ', counter)
            except:
                print('Unexpected error at line:', counter)
                raise
    return embeddings


def shrink_to_vocabulary(embeddings_input_path, vocabulary):
    embeddings = load(embeddings_input_path, vocabulary)
    dirname, filename = os.path.split(embeddings_input_path)
    outputdir = os.path.join(dirname, 'shrunk')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    embeddings_output_path = os.path.join(outputdir, filename)
    save(embeddings, embeddings_output_path)


def save(embeddings, path):
    with open(path, 'w') as f:
        for word in embeddings.keys():
            vector = embeddings[word]
            values = ' '.join(map(str, vector))
            f.write(word + ' ' + values + '\n')


def load_bin(path):
    return KeyedVectors.load_word2vec_format(path, binary=True)