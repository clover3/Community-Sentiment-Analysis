import numpy
from keras.engine import Input, merge, Model
from keras.layers import Embedding, Dropout, Convolution1D, MaxPooling1D, Flatten, Dense, Activation, Reshape, \
    Convolution2D, Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adadelta, SGD

from CNN import len_sentence, l2value, entity_embed_length


def get_simple_model(len_embedding, data, filter_sizes):
    dropout_prob = (0.1, 0.3)
    num_filters = 200
    hidden_dims = 100
    embedding_dim = len_embedding
    size_voca = len(data.idx2vect)

    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_level_input')

    sent_x = Embedding(size_voca, embedding_dim,
      input_length=len_sentence, weights=[data.idx2vect])(sent_input)


    sent_x = Dropout(dropout_prob[0], input_shape=(len_sentence, embedding_dim))(sent_x)

    multiple_filter_output= []
    for i in xrange(len(filter_sizes)):
        conv = Convolution1D(nb_filter=num_filters,
          filter_length= filter_sizes[i],
          border_mode='valid',
          bias=True,
          activation='relu',
          subsample_length=1)(sent_x)
        pool = MaxPooling1D(pool_length = len_sentence - filter_sizes[i] + 1)(conv)
        multiple_filter_output.append(Flatten()(pool))

    if len(filter_sizes) == 1:
        sent_v = multiple_filter_output[0]
    else:
        sent_v = merge(multiple_filter_output, mode = 'concat')

    sent_v = Dense(hidden_dims)(sent_v)
    sent_v = Dropout(dropout_prob[1])(sent_v)
    sent_v = Activation('relu')(sent_v)
    sent_loss = Dense(3, activation='softmax', name='sent_level_output')(sent_v)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=l2value)

    model = Model(input=sent_input, output=sent_loss)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adadelta)
    return model


def rnn2way(len_embedding, len_sentence, idx2vect):
    model = Sequential()
    size_voca = len(idx2vect)
    model.add(Embedding(size_voca, len_embedding, input_length=len_sentence))
    model.add(LSTM(100))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def entityMatchingRnn(len_embedding, len_sentence, data):
    num_class = 3

    size_voca = len(data.idx2vect)

    sent_model = Sequential()
    sent_model.add(Embedding(size_voca, len_embedding,
                       input_length=len_sentence, weights=[data.idx2vect]))
    sent_model.add(LSTM(100))

    num_entity = len(set(data.keyword2idx.keys()))
    eew = [numpy.random.uniform(-0.01, 0.01, size=(num_entity, entity_embed_length))]

    entity_model = Sequential()
    entity_model.add(Embedding(num_entity, entity_embed_length, input_length=1, weights=eew))
    entity_model.add(Reshape([entity_embed_length]))

    final_model = Sequential()
    final_model.add(Merge([sent_model, entity_model], mode='concat'))
    final_model.add(Dense(num_class, activation='softmax', name='sent_level_output'))

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, clipnorm=l2value)
    final_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adadelta)

    return final_model


def get2way_model(len_embedding, len_sentence, idx2vect, filter_sizes):
    dropout_prob = (0.1, 0.3)
    num_filters = 10
    hidden_dims = 10
    embedding_dim = len_embedding
    size_voca = len(idx2vect)

    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_level_input')

    sent_x = Embedding(size_voca, embedding_dim,
      input_length=len_sentence, weights=[idx2vect])(sent_input)


    sent_x = Dropout(dropout_prob[0], input_shape=(len_sentence, embedding_dim))(sent_x)

    multiple_filter_output= []
    for i in xrange(len(filter_sizes)):
        conv = Convolution1D(nb_filter=num_filters,
          filter_length= filter_sizes[i],
          border_mode='valid',
          bias=True,
          activation='relu',
          subsample_length=1)(sent_x)
        pool = MaxPooling1D(pool_length = len_sentence - filter_sizes[i] + 1)(conv)
        multiple_filter_output.append(Flatten()(pool))

    if len(filter_sizes) == 1:
        sent_v = multiple_filter_output[0]
    else:
        sent_v = merge(multiple_filter_output, mode = 'concat')

    sent_v = Dense(hidden_dims)(sent_v)
    sent_v = Dropout(dropout_prob[1])(sent_v)
    sent_v = Activation('relu')(sent_v)
    sent_loss = Dense(2, activation='softmax', name='sent_level_output')(sent_v)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=l2value)

    model = Model(input=sent_input, output=sent_loss)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy', 'fmeasure'], optimizer=adadelta)
    return model


def get_entity_masking_model(len_embedding, data, filter_sizes):
    num_class = 3
    num_filters = 200
    embedding_dim = len_embedding
    size_voca = len(data.idx2vect)
    num_entity = len(set(data.keyword2idx.keys()))
    eew = [numpy.random.uniform(-0.01, 0.01, size=(num_entity, entity_embed_length))]

    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_level_input')
    ei_input = Input(shape=(1,), name='entity_indicator_input')

    sent_x = Embedding(size_voca, embedding_dim,
      input_length=len_sentence, weights=[data.idx2vect])(sent_input)

    ei_emb = Embedding(num_entity, entity_embed_length, input_length=1, weights=eew)(ei_input)
    ei_emb = Reshape([entity_embed_length])(ei_emb)

    sent_x = Dropout(0.5, input_shape=(len_sentence, embedding_dim))(sent_x)
    ei_emb = Dropout(0.5, input_shape=(1, entity_embed_length))(ei_emb)

    multiple_filter_output= []
    for i in xrange(len(filter_sizes)):
        conv = Convolution1D(nb_filter=num_filters,
          filter_length= filter_sizes[i],
          border_mode='valid',
          bias=True,
          activation='relu',
          subsample_length=1)(sent_x)
        pool = MaxPooling1D(pool_length = len_sentence - filter_sizes[i] + 1)(conv)
        multiple_filter_output.append(Flatten()(pool))

    if len(filter_sizes) == 1:
        text_feature = multiple_filter_output[0]
    else:
        text_feature = merge(multiple_filter_output, mode = 'concat') # text features from CNN

    text_ei_feature = merge([text_feature, ei_emb], mode='concat')
    text_ei_feature = Dropout(0.5)(text_ei_feature)
    sent_loss = Dense(num_class, activation='softmax', name='sent_level_output')(text_ei_feature)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, clipnorm=l2value)

    model = Model(input=[sent_input, ei_input], output=sent_loss) # TODO : take multiple inputs
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adadelta)

    return model


def get_model(len_embedding, size_voca):
    print("Creating Model : ")
    n_filter = 500

    model = Sequential()
    model.add(Input(len_sentence))
    model.add(Embedding(size_voca, len_embedding, input_length=len_sentence))
    model.add(Reshape((1, len_sentence, len_embedding)))  ### [ len_sentence * len_embedding ]

    filter_size = 5
#    filter_sizes = [3,4,5]
#    convs = []
#    for filter_size in filter_sizes:
#        convs.append( Convolution1D(n_filter, filter_size, init="uniform", activation='relu', bias=True) )

#    model.add(Convolution1D(n_filter, 5, init="uniform", activation='relu', bias=True)) ###  [len_sentence *
    model.add(Convolution2D(n_filter, len_embedding, filter_size, init="uniform", activation='relu', bias=True)) ###  [len_sentence *
    #model.add(Activation("relu"))
    model.add(MaxPooling1D(len_sentence - filter_size + 1) )
    model.add(Dropout(0.1))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model