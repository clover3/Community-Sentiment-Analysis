# -*- coding: utf-8 -*-
import os
#os.environ['THEANO_FLAGS'] = 'device=gpu1'

from clover_lib import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Embedding, Reshape, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from sklearn.preprocessing import LabelEncoder
import cPickle
import numpy
import os
import random

seed = 7
numpy.random.seed(seed)

len_sentence = 80

"""
predefined words : 0 : padding 1 : keyword marker

"""

def load_vec(fname, vocab, binary = True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("  Loading word2vec...")
    #w2v_cache = "cache\\w2v"
    #if os.path.isfile(w2v_cache):
    #    return cPickle.load(open(w2v_cache,"rb"))

    mode = ("rb" if binary else "r")
    word_vecs = {}
    with open(fname, mode) as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * layer1_size

        def getline():
            if binary:
                return numpy.fromstring(f.read(binary_len), dtype='float32')
            else:
                return numpy.array(f.readline().split(), dtype='float32')

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = getline()
            else:
                getline()
    print("  Loaded word2vec...")
#    cPickle.dump(word_vecs, open(w2v_cache, "wb"))
    return word_vecs


def get_voca(text_list):
    voca = set()
    for text in text_list:
        voca.update(text.split())
    return voca


def build_index(voca, w2v, k):
    print("building index..")
    predefined_word = 5
    index = predefined_word
    word2idx = dict()
    idx2vect = numpy.zeros(shape=(len(voca) + predefined_word, k), dtype='float32')

    for i in range(predefined_word):
        #idx2vect[i] = numpy.zeros(k, dtype='float32')
        idx2vect[i] = numpy.random.uniform(-0.25,0.25,k)

    f = open("missing_w2v.txt", "w")
    if w2v is not None:
        for word in w2v.keys():
            word2idx[word] = index
            idx2vect[index] = w2v[word]
            index += 1

    match_count = index

    for word in voca:
        if word not in word2idx:
            f.write(word + "\n")
            word2idx[word] = index
            idx2vect[index] = numpy.zeros(k, dtype='float32')
            index += 1
    f.close()

    print("w2v {} of {} matched".format(match_count, index))
    return word2idx, idx2vect


class DataSet:
    def __init__(self, dataset_x, data_set_e, dataset_y, idx2vect, len_embedding, keywords2idx):
        data = zip(dataset_x, data_set_e, dataset_y)
        numpy.random.shuffle(data)
        len_train = int(len(data) * 0.9)
        train_x, train_e, train_y = zip(*data[0:len_train])
        test_x, test_e, test_y = zip(*data[len_train:])

        self.train_x = numpy.array(train_x)
        self.train_e = numpy.array(train_e)
        self.train_y = numpy.array(train_y)
        self.test_x = numpy.array(test_x)
        self.test_e = numpy.array(test_e)
        self.test_y = numpy.array(test_y)

        self.keyword2idx = keywords2idx
        self.idx2vect = idx2vect
        self.len_embedding = len_embedding

def  tokenize(list_str):
    from konlpy.tag import Twitter
    from konlpy.tag import Hannanum
    lib = Twitter()

    arr = []
    for sentence in list_str:
        tokens = map(lambda x:x[0].encode('utf8'), lib.pos(sentence))
        arr.append(tokens)
    return arr


def load_data_agree(neu_path, pos_path, neg_path, w2v_path, dimension):
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
    dataset_y = len(raw_neu) * [0] + len(raw_pos) * [1] + len(raw_neg) * [2]

    return DataSet(dataset_x, dataset_y, idx2vect, dimension)


def load_data_get_word2idx(label_path, path_article, w2v_path, dimension):
    IDX_R_THREAD = 1
    IDX_R_ARTICLE = 2
    IDX_R_KEYWORD= 3
    IDX_R_LABEL = 4
    # load label table
    labels = load_csv(label_path)
    labels = [label for label in labels if label[IDX_R_LABEL] == '1' or label[IDX_R_LABEL] == '3' or label[IDX_R_LABEL] == '2' ]
    n_pos =  len([label for label in labels if label[IDX_R_LABEL] == '1'])
    n_neg = len([label for label in labels if label[IDX_R_LABEL] == '3'])
    # load article table
    articles = load_csv(path_article)
    articles = parse_token(articles)
    article_dic = dict()
    for article in articles :
        article_id = article[IDX_ARTICLE_ID]
        thread_id = article[IDX_THREAD_ID]
        article_dic[(article_id,thread_id)] = article

    voca = set(flatten([article[IDX_TOKENS] for article in articles]))

    w2v = load_vec(w2v_path, voca, False)
    word2idx, idx2vect = build_index(voca, w2v, dimension)
    return word2idx


def set2idxmap(s):
    d = dict()
    cnt = 0
    for item in s :
        d[item] = cnt
        cnt += 1
    return d

def load_data_label(label_path, path_article, w2v_path, dimension):
    IDX_R_THREAD = 1
    IDX_R_ARTICLE = 2
    IDX_R_KEYWORD= 3
    IDX_R_LABEL = 4
    print("Loading Data...")
    # load label table
    labels = load_csv(label_path)
    labels = [label for label in labels if label[IDX_R_LABEL] == '1' or label[IDX_R_LABEL] == '3' or label[IDX_R_LABEL] == '2' ]
    labels_pos = [label for label in labels if label[IDX_R_LABEL] == '1']
    labels_neu = [label for label in labels if label[IDX_R_LABEL] == '2']
    labels_neg = [label for label in labels if label[IDX_R_LABEL] == '3']

    labels = labels_pos + labels_neu + labels_neg
    random.shuffle(labels)

    all_keyword = set([label[IDX_R_KEYWORD] for label in labels])
    keyword2idx = set2idxmap(all_keyword)

    n_pos =  len([label for label in labels if label[IDX_R_LABEL] == '1'])
    n_neu =  len([label for label in labels if label[IDX_R_LABEL] == '2'])
    n_neg = len([label for label in labels if label[IDX_R_LABEL] == '3'])
    print("  Corpus size : positive={} negative={} neutral={}...".format(n_pos, n_neg, n_neu) )
    print("  Loading articles...")
    # load article table
    articles = load_csv(path_article)
    print("  parsing tokens...")
    articles = parse_token(articles)
    article_dic = dict()
    for article in articles :
        article_id = article[IDX_ARTICLE_ID]
        thread_id = article[IDX_THREAD_ID]
        article_dic[(article_id,thread_id)] = article

    voca = set(flatten([article[IDX_TOKENS] for article in articles]))

    w2v = load_vec(w2v_path, voca, False)
    print("  Loaded w2v len = {}".format(len(w2v)) )
    word2idx, idx2vect = build_index(voca, w2v, dimension)

    def indexize(label):
        def find_keyword(article, keyword):
            tokens = article[IDX_TOKENS]
            indexs = [i for i,x in enumerate(tokens) if (keyword in x)]
            if len(indexs) > 0:
                return indexs[0]
            # if keyword is mis-splitted, we need to find its location.
            ### IT CONTAINS BUG! ###

            scan = ""
            for i, token in list(enumerate(tokens))[::-1]:
                scan = token + scan
                if keyword.lower() in scan.lower():
                    return i

            return 0
            print(keyword)
            print(article[IDX_CONTENT])
            print(scan)
        #  lookup the corpus to find specified article

        article = article_dic[(label[IDX_R_ARTICLE], label[IDX_R_THREAD])]
        keyword = label[IDX_R_KEYWORD]
        idx_keyword = find_keyword(article, keyword)
        pre_len = (len_sentence-1) / 2
        post_len = len_sentence - pre_len - 1

        # slice the content to specific size.
        # If there isn't enough token before keyword, use all from beginning
        if idx_keyword < pre_len :
            begin = 0
            end = len_sentence
        else :
            begin = idx_keyword - pre_len
            end = idx_keyword - pre_len + len_sentence
        #print "Article[{}]/[{}] [{}]:".format(label[IDX_R_ARTICLE], label[IDX_R_THREAD],label[IDX_R_LABEL])

        # if lenght exceed, pad with zero
        tokens = article[IDX_TOKENS]
        result = []
        for i in range(begin, end):
            if i < len(tokens) :
                result.append(word2idx[tokens[i]])
            else :
                result.append(0)
        return result

    def label2x(label):
        index = indexize(label)
        keyword = keyword2idx[label[IDX_R_KEYWORD]]
        return numpy.array([index, keyword])

    data_set_x = numpy.array([indexize(label) for label in labels])
    data_set_e = numpy.array([ keyword2idx[label[IDX_R_KEYWORD]] for label in labels])
    def label2hotv(label):
        if label[IDX_R_LABEL] == '3':
            return (1,0,0)
        elif label[IDX_R_LABEL] == '2':
            return (0,1,0)
        elif label[IDX_R_LABEL] == '1':
            return (0,0,1)
        else:
            raise Exception("Not expected")


    data_set_y = [label2hotv(label) for label in labels]

    return DataSet(data_set_x, data_set_e, data_set_y, idx2vect, dimension, keyword2idx)


def load_data(pos_path, neg_path, w2v_path, dimension):
    print("Loading Data...")
    raw_pos = open(pos_path).readlines()
    raw_neg = open(neg_path).readlines()
    raw_data = raw_pos+raw_neg
    w2v = load_vec(w2v_path, get_voca(raw_data)) # w2v : word->float[]

    voca = get_voca(raw_data)
    word2idx, idx2vect = build_index(voca, w2v, dimension)

    # Convert text into index
    vectors = []
    for text in raw_data:
        vector = list(map(lambda x: word2idx[x], text.split()))
        while len(vector) < len_sentence :
            vector.append(0)
        vectors.append(numpy.array(vector, dtype='float32'))

    dataset_x = numpy.array(vectors)
    dataset_y = len(raw_pos)*[1] + len(raw_neg) * [0]
    return DataSet(dataset_x, dataset_y, idx2vect, dimension)


def load_data_gt(pos_path, neg_path, w2v_path, dimension):
    print("Loading Data...")
    raw_pos = codecs.open(pos_path, "r", "cp949").readlines()
    raw_neg = codecs.open(neg_path, "r", "cp949").readlines()
    raw_data = raw_pos+raw_neg
    token_data = tokenize(raw_data)
    print("  Corpus size : positive={} negative={} ".format(len(raw_pos), len(raw_neg)))

    for token in token_data[0]:
        print token,

    voca = set(flatten(token_data))
    w2v = load_vec(w2v_path, voca, False) # w2v : word->float[]
    word2idx, idx2vect = build_index(voca, w2v, dimension)

    pickle.dump(word2idx, open("word2idx","wb"))
    # Convert text into index
    vectors = []
    for tokens in token_data:
        vector = list(map(lambda x: word2idx[x], tokens[:len_sentence]))
        while len(vector) < len_sentence :
            vector.append(0)
        vectors.append(numpy.array(vector, dtype='float32'))

    dataset_x = numpy.array(vectors)
    dataset_y = len(raw_pos)*[(1,0)] + len(raw_neg) * [(0,1)]
    return DataSet(dataset_x, dataset_y, idx2vect, dimension)


def load_data_carsurvey(path_label, w2v_path, dimension):
    ## modelName,fault,summary,content,label,modelYear,date,buyAnother,manufacturer
    data = load_csv(path_label)
    data = filter(lambda x: x[4]!="neutral", data)
    raw_data = [line[2] for line in data]

    w2v = load_vec(w2v_path, get_voca(raw_data), False)  # w2v : word->float[]

    voca = get_voca(raw_data)
    word2idx, idx2vect = build_index(voca, w2v, dimension)
    # Convert text into index

    vectors = []
    for text in raw_data:
        vector = list(map(lambda x: word2idx[x], text.split()))
        while len(vector) < len_sentence:
            vector.append(0)
        vector = vector[:len_sentence]
        vectors.append(numpy.array(vector, dtype='float32'))

    dataset_x = numpy.array(vectors)
    def text2label(text):
        if text == "positive":
            return 2
        elif text== "negative":
            return 0
        elif text == "neutral":
            return 1
        else:
            raise "Unexpected"

    dataset_y = [ text2label(line[4]) for line in data]
    return DataSet(dataset_x, dataset_y, idx2vect, dimension)


entity_embed_length = 50
def buildModel(len_embedding, data, filter_sizes):
    num_class = 3
    num_filters = 500
    embedding_dim = len_embedding
    size_voca = len(data.idx2vect)
    numentity = len(set(data.keyword2idx.keys()))
    eew = [numpy.random.uniform(-0.01, 0.01, size=(numentity, entity_embed_length))]

    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_level_input')
    ei_input = Input(shape=(1,), name='entity_indicator_input')

    sent_x = Embedding(size_voca, embedding_dim,
      input_length=len_sentence, weights=[data.idx2vect])(sent_input)

    ei_emb = Embedding(numentity, entity_embed_length, input_length=1, weights=eew)(ei_input)
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
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

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


def get_model_dirty(len_embedding, data, filter_sizes):
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
    sent_loss = Dense(3, activation='sigmoid', name='sent_level_output')(sent_v)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

    model = Model(input=sent_input, output=sent_loss)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adadelta)
    return model

def run_english():
    len_embedding = 300

    path_prefix = "..\\input\\rt-polarity."
    pos_path = path_prefix + "pos"
    neg_path = path_prefix + "neg"
    w2v_path = "..\\input\\GoogleNews-vectors-negative300.bin"

    data = load_data(pos_path, neg_path, w2v_path, len_embedding)

    model = get_model_dirty(len_embedding, data, [3, 4, 5])
    #model = get_model(len_embedding, data.voca_size)

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=50,
              nb_epoch=40,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)

def run_carsurve():
    path_lable = "..\\input\\carsurvey.csv"
    ## modelName,fault,summary,content,label,modelYear,date,buyAnother,manufacturer
    len_embedding = 50

    w2v_path = "..\\input\\english_word2vec_50_vectors.txt"

    data = load_data_carsurvey(path_lable, w2v_path, len_embedding)

    model = get_model_dirty(len_embedding, data, [3, 4, 5])

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=200,
              nb_epoch=40,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)


def run_gt():
    len_embedding = 50

    pos_path = "input\\pos.txt"
    neg_path = "input\\neg.txt"
    w2v_path = "D:\\data\\input\\korean_word2vec_wv_50.txt"

    data = load_data_gt(pos_path, neg_path, w2v_path, len_embedding)

    model = get_model_dirty(len_embedding, data, [3,4,5,6,7,8,9,10,11,12,13])
    #model = get_model(len_embedding, data.voca_size)

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=100,
              nb_epoch=50,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)

    return model


def run_korean():
    len_embedding = 50

    path_lable = "D:\\data\\corpus_samba.csv"
    w2v_path = "D:\\data\\input\\korean_word2vec_wv_50.txt"
    path_article = "D:\\data\\input\\bobae_car_tkn_twitter.csv"
    #path_article = "D:\\data\\input\\babae_car_recovered2.csv"

    data = load_data_label(path_lable, path_article, w2v_path, len_embedding)

    #model = get_model_dirty(len_embedding, data, [3,4,5,6,7])
    model = buildModel(len_embedding, data, [3,4,5,6,7])

    print("Training Model...")
    model.fit([data.train_x, data.train_e], data.train_y,
              batch_size=300,
              nb_epoch=300,
              validation_data=([data.test_x, data.test_e], data.test_y),
              shuffle=True)

    return model


def run_agreement():
    len_embedding = 50

    path_prefix = "..\\input\\agree_"
    neu_path = path_prefix + "0.csv"
    pos_path = path_prefix + "1.csv"
    neg_path = path_prefix + "2.csv"
    w2v_path = "..\\input\\korean_word2vec_wv_50.txt"

    data = load_data_agree(neu_path, pos_path, neg_path, w2v_path, len_embedding)

    model = get_model_dirty(len_embedding, data, [3,4,5,6,7,8,9])
    #model = get_model(len_embedding, data.voca_size)

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=50,
              nb_epoch=100,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)



import pickle
def save_word2idx():
    len_embedding = 50

    path_lable = "D:\\data\\corpus_samba.csv"
    w2v_path = "D:\\data\\input\\korean_word2vec_wv_50.txt"
    path_article = "D:\\data\\input\\bobae_car_tkn_twitter.csv"
    word2idx = load_data_get_word2idx(path_lable, path_article, w2v_path, len_embedding)

    pickle.dump(word2idx, open("word2idx","wb"))

if __name__ == "__main__":
    #save_word2idx()
    #model = run_gt()
    #model.save("model")
    #run_agreement()
    #run_carsurve()
    run_korean()
