from __future__ import print_function

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import eswc2018_dataset_handler as dataset
import eswc2018_word_embeddings as eswc18we
from preprocessing import extract_text, polarity_to_int
import settings
import pickle
import os

np.random.seed(2111)

data = dataset.read(os.path.join(settings.DATA_PATH, 'Music'))
np.random.shuffle(data)
test_data = dataset.read_xml(settings.TEST_PATH)

we_size = 256
we_epochs = 50

print('Loading word embeddings...')
embeddings = eswc18we.load(we_size, we_epochs)

print('Word Embeddings size:', we_size)
print('Word Embeddings epochs:', we_epochs)
print()

VALIDATION_SPLIT = 0.1

texts = [extract_text(record) for record in data]
polarities = [record['polarity'] for record in data]
labels = [polarity_to_int(polarity) for polarity in polarities]
texts_test = [extract_text(record) for record in test_data]

print('Tokenizing...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts + texts_test)
sequences = tokenizer.texts_to_sequences(texts)
sequences_test = tokenizer.texts_to_sequences(texts_test)
lengths = [len(seq) for seq in sequences]
max_length = min(max(lengths), 256)

word_index = tokenizer.word_index

data = pad_sequences([x for x in sequences if x], maxlen=max_length)
test = pad_sequences(sequences_test, maxlen=max_length)
labels = to_categorical(np.asarray(labels))

validation_size = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-validation_size]
y_train = labels[:-validation_size]
x_val = data[-validation_size:]
y_val = labels[-validation_size:]

training = x_train, y_train
validation = x_val, y_val

num_words = len(word_index) + 1
print('num_words', num_words)
embeddings_matrix = np.zeros((num_words, we_size))
for word, i in word_index.items():
    vector = embeddings.get(word)
    if vector is not None:
        embeddings_matrix[i] = vector

print('Saving data...')
with open('training.pickle', 'wb') as f:
    pickle.dump(training, f)

with open('validation.pickle', 'wb') as f:
    pickle.dump(validation, f)

with open('test.pickle', 'wb') as f:
    pickle.dump(test, f)

embeddings_matrix_path = 'embeddings_matrix_%d_%d.npy' % (we_size, we_epochs)
np.save(embeddings_matrix_path, embeddings_matrix)
