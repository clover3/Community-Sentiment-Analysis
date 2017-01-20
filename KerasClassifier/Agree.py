
from CNN_common import *


from clover_lib import *
from keras.models import Sequential
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Embedding, Reshape, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adadelta
import numpy
import random

class Agree_DataSet:
    def __init__(self, dataset_x, dataset_y, idx2vect, len_embedding):
        data = zip(dataset_x, dataset_y)
        numpy.random.shuffle(data)
        len_train = int(len(data) * 0.9)
        train_x, train_y = zip(*data[0:len_train])
        test_x, test_y = zip(*data[len_train:])

        self.train_x = numpy.array(train_x)
        self.train_y = numpy.array(train_y)
        self.test_x = numpy.array(test_x)
        self.test_y = numpy.array(test_y)

        self.idx2vect = idx2vect
        self.len_embedding = len_embedding

def load_data_agree(neu_path, pos_path, neg_path, w2v_path, dimension, len_sentence):
    IDX_R_THREAD = 1
    IDX_R_ARTICLE = 2
    IDX_R_KEYWORD= 3
    IDX_R_LABEL = 4
    print("Loading Data...")
    # load label table
    raw_neu = codecs.open(neu_path, "r",encoding="euc-kr", errors="replace").readlines()
    raw_pos = codecs.open(pos_path, "r",encoding="euc-kr", errors="replace").readlines()
    raw_neg = codecs.open(neg_path, "r",encoding="euc-kr", errors="replace").readlines()
    raw_data = raw_neu +raw_pos+raw_neg
    print("  Corpus size : neutral={} positive={} negative={}...".format(len(raw_neu),len(raw_pos),len(raw_neg)) )
    print("  Loading articles...")
    # load article table
    print("  parsing tokens...")
    tokened_data = tokenize(raw_data)

    voca = set(flatten(tokened_data))
    print(voca)
    print("-----------")

    w2v = load_vec(w2v_path, voca, False)
    print("  Loaded w2v len = {}".format(len(w2v)) )
    word2idx, idx2vect = build_index(voca, w2v, dimension)

    # Convert text into index
    vectors = []
    for tokens in tokened_data:
        vector = list(map(lambda x: word2idx[x], tokens))
        vector = vector[:len_sentence]
        while len(vector) < len_sentence:
            vector.append(0)
        vectors.append(numpy.array(vector, dtype='float32'))

    dataset_x = numpy.array(vectors)
    dataset_y = len(raw_neu) * [[0,0]]+ len(raw_pos) * [[1,0]] + len(raw_neg) * [[0,1]]

    return Agree_DataSet(dataset_x, dataset_y, idx2vect, dimension)


def get_model_agree(len_sentence, len_embedding, data, filter_sizes):
    print("Creating Model... ")

    dropout_prob = (0.2, 0.2)
    n_filter = 200
    hidden_dims = 100
    size_voca = len(data.idx2vect)
    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_input')
    print(" Embedding Layer ")
    sent_x = Embedding(size_voca, len_embedding,
                       input_length=len_sentence, weights=[data.idx2vect])(sent_input)

    sent_x = Dropout(dropout_prob[0], input_shape=(len_sentence, len_embedding))(sent_x)
    multiple_filter_output= []

    convs = []
    pools = []
    for i, filter_size in enumerate(filter_sizes):
        convs.append(Convolution1D(nb_filter=n_filter,
                         filter_length= filter_size,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(sent_x))
        pools.append(MaxPooling1D(pool_length = len_sentence - filter_size + 1)(convs[i]))
        multiple_filter_output.append(Flatten()(pools[i]))

    sent_v = merge(multiple_filter_output, mode = 'concat')

    sent_v = Dense(hidden_dims)(sent_v)
    sent_v = Dropout(dropout_prob[1])(sent_v)
    sent_v = Activation('relu')(sent_v)
    sent_loss = Dense(2, activation='sigmoid', name='sent_level_output')(sent_v)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

    model = Model(input=sent_input, output=sent_loss)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adadelta)
    return model

def run_agreement():
    len_embedding = 50
    len_sentence = 80

    path_prefix = "..\\input\\agree_"
    neu_path = path_prefix + "0.csv"
    pos_path = path_prefix + "1.csv"
    neg_path = path_prefix + "2.csv"
    w2v_path = "..\\input\\korean_word2vec_wv_50.txt"

    data = load_data_agree(neu_path, pos_path, neg_path, w2v_path, len_embedding, len_sentence)

    model = get_model_agree(len_sentence, len_embedding, data, [2,3,4,5])

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=200,
              nb_epoch=200,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)


if __name__ == "__main__":
    run_agreement()