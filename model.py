from __future__ import print_function

from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout, Bidirectional, LSTM, \
    GaussianNoise, Activation, MaxoutDense
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import preprocessing

from attention import Attention


def build_embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                           scale=False, normalize=False):
    if scale:
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return layer


def build_rnn(unit=LSTM, cells=64, bi=False, return_sequences=True, recurrent_dropout=0., l2_reg=0):
    rnn = unit(cells,
               return_sequences=return_sequences,
               recurrent_dropout=recurrent_dropout,
               kernel_regularizer=l2(l2_reg),
               implementation=1)
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def build_model(embeddings, classes, max_length, unit=LSTM, cells=64,
                layers=1, train_embeddings=False, attention=False, **kwargs):
    bidirectional = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_embeddings = kwargs.get("dropout_embeddings", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    recurrent_dropout = kwargs.get("recurrent_dropout", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)

    model = Sequential()
    model.add(build_embeddings_layer(max_length=max_length, embeddings=embeddings,
                                     trainable=train_embeddings, masking=True, scale=False,
                                     normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_embeddings > 0:
        model.add(Dropout(dropout_embeddings))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        rnn = build_rnn(unit, cells, bidirectional, return_sequences=True,
                        recurrent_dropout=recurrent_dropout)
        model.add(rnn)
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention:
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model
