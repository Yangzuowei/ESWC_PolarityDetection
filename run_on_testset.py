from __future__ import print_function

import numpy
import numpy as np
from keras.layers import LSTM

import eswc2018_dataset_handler as dataset
import settings
from model import build_model
import pickle
from preprocessing import int_to_polarity

test_data = dataset.read_xml(settings.TEST_PATH)
ids_test = [record['@id'] for record in test_data]

we_size = 256
we_epochs = 50

print('Loading pre-processed test data...')
with open('test.pickle', 'rb') as f:
    test = pickle.load(f)

print('Loading word embeddings...')
embeddings_matrix_path = 'embeddings_matrix_%d_%d.npy' % (we_size, we_epochs)
embeddings_matrix = np.load(embeddings_matrix_path)

model = build_model(embeddings_matrix, attention=True, classes=2,
                    max_length=256, layers=2, unit=LSTM,
                    cells=64, bidirectional=True,
                    train_embeddings=True,
                    noise=0.3, clipnorm=1, lr=0.001, loss_l2=0.0001,
                    final_layer=False, dropout_final=0.5,
                    dropout_attention=0.5,
                    dropout_embeddings=0.3, dropout_rnn=0.3,
                    recurrent_dropout=0.3)

model.load_weights('BiLSTM-attention-final-fine-tuning-e02-acc-0.956.hdf5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

print('Computing predictions...')
with open('result.txt', 'w') as f:
    for i in range(0, len(ids_test)):
        current_id = ids_test[i]
        seq = np.array(np.array([test[i]]))
        label = model.predict_classes(seq)[0]
        polarity = int_to_polarity(label)
        f.write(current_id + ';' + polarity + '\n')
