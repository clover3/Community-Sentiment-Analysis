# -*- coding: utf-8 -*-

from CNN_common import load_vec, build_index, tokenize_list, tokenize

from config import *
from clover_lib import *
from SA_Label import *
from Models import *
import pickle

from keras.models import load_model
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Embedding, Reshape, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adadelta
from sklearn.model_selection import StratifiedKFold
import numpy



class DataSet2Way:
    def __init__(self, dataset_x, dataset_y, idx2vect, len_embedding, word2idx):
        data = zip(dataset_x, dataset_y)
        #numpy.random.shuffle(data)
        x, y = zip(*data)

        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.idx2vect = idx2vect
        self.word2idx = word2idx
        self.len_embedding = len_embedding


def get_voca(texts):
    return set(flatten(tokenize_list(texts)))


def load_data(pos_path, neg_path, w2v_path, len_embedding, len_sentence):
    print("Loading Data...")
    raw_pos = open(pos_path).readlines()[400:]
    raw_neg = open(neg_path).readlines()[2000:]

    raw_data = raw_pos+raw_neg
    voca = get_voca(raw_data)
    token_list = tokenize_list(raw_data)

    w2v = load_vec(w2v_path, voca, False) # w2v : word->float[]

    word2idx, idx2vect = build_index(voca, w2v, len_embedding)

    # Convert text into index
    vectors = []
    for tokens in token_list:
        vector = list(map(lambda x: word2idx[x], tokens))[:len_sentence]
        while len(vector) < len_sentence :
            vector.append(0)
        vectors.append(numpy.array(vector, dtype='float32'))

    dataset_x = numpy.array(vectors)
    dataset_y = len(raw_pos)*[(1,0)] + len(raw_neg) * [(0,1)]
    return DataSet2Way(dataset_x, dataset_y, idx2vect, len_embedding, word2idx)

def run_2way():
    len_embedding = 100
    len_sentence = 80

    load_pickle = config_load_pickle_for_data
    if load_pickle:
        data = pickle.load(open(config_pickle_load_path, "rb"))
        print("Loading data from pickle")
    else:
        path_pos = "..\\input\\car_context.txt"
        path_neg = "..\\input\\not_car_context.txt"
        w2v_path = "..\\input\\korean_word2vec_wv_100.txt"
        data = load_data(path_pos, path_neg, w2v_path, len_embedding, len_sentence)
        pickle.dump(data, open(config_pickle_save_path, "wb"))

    #model = get2way_model(len_embedding, len_sentence, data.idx2vect, [3, 4, 5])
    model = rnn2way(len_embedding, len_sentence, data.idx2vect)
    init_weight = model.get_weights()

    X = data.x
    y = data.y

    def split(X,y, validate_size):
        total_size = len(X)

        X_valid = numpy.concatenate((X[0:validate_size], X[-validate_size:]), 0)
        y_valid = numpy.concatenate((y[0:validate_size], y[-validate_size:]), 0)

        n_pos = sum(item[0] for item in y_valid )
        n_neg = sum(item[1] for item in y_valid)

        print("valid : pos={} neg={}".format(n_pos, n_neg))

        X_train = X[validate_size: -validate_size]
        y_train = y[validate_size: -validate_size]
        return X_train, y_train, X_valid, y_valid

    X_train, y_train, X_valid, y_valid = split(X,y, 1000)
    print("Training Model...")
    model.set_weights(init_weight)
    hist = model.fit(X_train, y_train,
                     batch_size=config_batch_size,
                     nb_epoch=config_epoch,
                     validation_data=(X_valid, y_valid),
                     shuffle=True)
    print("\nAccuracy : %f (at %d)\n" % (max(hist.history['val_acc']), numpy.argmax(hist.history['val_acc'])))
    print max(hist.history['val_acc'])

    return model, data.idx2vect, data.word2idx

def convert2index(tokens, word2idx):
    def translate(x):
        if x in word2idx:
            return word2idx[x]
        else:
            return 1

    vector = list(map(lambda x: translate(x), tokens))[:len_sentence]
    while len(vector) < len_sentence :
        vector.append(0)
    return numpy.array(vector, dtype='float32')

def load_entity_corpus():
    path = "..\\input\\EntityLabel_utf.csv"
    EL_IDX_CONTENT     = 3
    EL_IDX_LABEL       = 4
    return [(line[EL_IDX_CONTENT],line[EL_IDX_LABEL]) for line in load_csv(path)]


def test_corpus(model, word2idx):

    def predict(model, sentence, entity):
        tokens = tokenize(sentence)
        e_sentence = convert2index(tokens, word2idx)
        X = numpy.array([e_sentence])
        return numpy.argmax(model.predict(X))

    corpus = load_entity_corpus()

    def calc(label):
        if label == "-":
            return 1
        else:
            return 0

    acc = 0
    total = 0
    for (sentence,label) in corpus:
        r = predict(model, sentence, word2idx)
        print r, label

        total +=1
        if r == calc(label):
            acc += 1

    print("{}/{}".format(acc,total))

if __name__ == "__main__":
    model, idx2vect, word2idx = run_2way()
    #model.save("model.h5")
    #pickle.dump(word2idx, open("trained2way.p", "wb"))
    #word2idx = pickle.load(open("trained2way.p", "rb"))
    #model = load_model("model.h5")
    #test_corpus(model, word2idx)
   #play_process_completed()