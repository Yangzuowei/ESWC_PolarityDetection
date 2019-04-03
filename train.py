from __future__ import print_function

import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from callbacks import WeightsCallback, ModelCheckpointOnBatch
from model import build_model

FINE_TUNING = False
we_size = 256
we_epochs = 50

print('Loading pre-processed data')

print('Loading training data...')
with open('training.pickle', 'rb') as f:
    training = pickle.load(f)

print('Loading validation data...')
with open('validation.pickle', 'rb') as f:
    validation = pickle.load(f)

print('Loading word embeddings...')
embeddings_matrix_path = 'embeddings_matrix_%d_%d.npy' % (we_size, we_epochs)
embeddings_matrix = np.load(embeddings_matrix_path)

model = build_model(embeddings_matrix, attention=True, classes=2,
                    max_length=256, layers=2,
                    bidirectional=True,
                    train_embeddings=FINE_TUNING,
                    noise=0.2, clipnorm=1, lr=0.001, loss_l2=0.0001,
                    final_layer=False, dropout_final=0.5,
                    dropout_attention=0.5,
                    dropout_embeddings=0.3, dropout_rnn=0.3,
                    recurrent_dropout=0.3)

if FINE_TUNING:
    model.load_weights('BiLSTM-attention-final-e02-acc-0.936.hdf5')


print('Plotting model to model.png')
plot_model(model, show_layer_names=False, show_shapes=True,
           to_file="model.png")

print(model.summary())

best_model_path_epoch = "BiLSTM-attention-final-e{epoch:02d}-acc-{val_acc:.3f}.hdf5"
# % (we_size, we_epochs)

best_model_path_batch = "BiLSTM-attention-final-period-checkpoint.hdf5"
# % (we_size, we_epochs)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=best_model_path_epoch,
                               verbose=1, save_best_only=False)

checkpointer_batch = ModelCheckpointOnBatch(filepath=best_model_path_batch,
                                            monitor='acc', mode='max',
                                            period=200,
                                            verbose=1, save_best_only=False)

weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])
callbacks = [checkpointer, checkpointer_batch, weights]

model.fit(training[0], training[1],
          validation_data=validation,
          epochs=2, batch_size=128,
          callbacks=callbacks)
