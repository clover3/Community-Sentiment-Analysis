# -*- coding: utf-8 -*-

from keras.layers import Convolution1D, MaxPooling1D, Embedding
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adadelta
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from CNN_common import *
from KoreanNLP import *
from Models import *
from SA_Label import *
from clover_lib import *
from config import *

numpy.random.seed(12)
len_sentence = 80
l2value = 2

"""
predefined words : 0 : padding 1 : keyword marker

"""

def load_vec(fname, vocab, binary = True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("Loading word2vec...")
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
    print("Loaded word2vec...")
#    cPickle.dump(word_vecs, open(w2v_cache, "wb"))
    return word_vecs


def get_voca(text_list):
    voca = set()
    for text in text_list:
        voca.update(text.split())
    return voca


class DataSet_validate:
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

class DataSet:
    def __init__(self, dataset_x, dataset_y, idx2vect, word2idx, len_embedding):
        data = zip(dataset_x, dataset_y)
        numpy.random.shuffle(data)
        x, y = zip(*data)

        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.idx2vect = idx2vect
        self.word2idx = word2idx
        self.len_embedding = len_embedding



class EMDataSet:
    def __init__(self, dataset_x, data_set_e, dataset_y, idx2vect, word2idx, len_embedding, keywords2idx):
        data = zip(dataset_x, data_set_e, dataset_y)
        numpy.random.shuffle(data)
        x, e, y = zip(*data)

        self.x = numpy.array(x)
        self.e = numpy.array(e)
        self.y = numpy.array(y)

        self.keyword2idx = keywords2idx
        self.idx2vect = idx2vect
        self.word2idx = word2idx
        self.len_embedding = len_embedding


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


def label2hotv(label):
    if label[IDX_R_LABEL] == '3':
        return (1, 0, 0)
    elif label[IDX_R_LABEL] == '2':
        return (0, 1, 0)
    elif label[IDX_R_LABEL] == '1':
        return (0, 0, 1)
    else:
        raise Exception("Not expected")


def get_article_dic(articles):
    article_dic = dict()
    for article in articles:
        article_id = article[IDX_ARTICLE_ID]
        thread_id = article[IDX_THREAD_ID]
        article_dic[(article_id, thread_id)] = article
    return article_dic

def all_voca(articles):
    def getcontent(article):
        return article[IDX_TITLE] + ". " + article[IDX_CONTENT]

    load_pickle = True
    if load_pickle:
        result_u = load_list("voca.txt")
        result = [str(e.encode("utf-8")) for e in result_u]
    else:
        texts = [getcontent(article) for article in articles]
        result =  set(flatten(tokenize_list(texts)))
        save_list(list(result), "voca.txt")

    return result


def convert2index(tokens, word2idx):
    result = []
    for token in tokens[:len_sentence]:
        try:
            result.append(word2idx[token])
        except KeyError as e:
            print "key not found {} (type={})".format(token, type(token))
            result.append(0)

    while len(result) < len_sentence:
        result.append(0)
    return result


def load_data_common(label_path, len_embedding, path_article, w2v_path):
    print("Loading Data...")
    # load label table
    labels = load_label(label_path)
    articles = load_csv(path_article)
    voca = all_voca(articles)
    w2v = load_vec(w2v_path, voca, False)
    print("  Loaded w2v len = {}".format(len(w2v)))
    word2idx, idx2vect = build_index(voca, w2v, len_embedding)
    article_dic = get_article_dic(articles)
    return word2idx, idx2vect, labels, article_dic


def load_data_sa(label_path, path_article, w2v_path, len_embedding):

    word2idx, idx2vect, labels, article_dic = load_data_common(label_path, len_embedding, path_article, w2v_path)

    token_dic = dict()

    def get_token(label):
        id = (label[IDX_R_ARTICLE], label[IDX_R_THREAD])
        if id in token_dic:
            return token_dic[id]
        else :
            article = article_dic[(label[IDX_R_ARTICLE], label[IDX_R_THREAD])]
            content = get_content(article)
            tokens = tokenize(content)
            token_dic[id] = tokens
            return tokens

    def extract_x(label):
        tokens = get_token(label)
        return convert2index(tokens, word2idx)

    def is_short(label):
        tokens = get_token(label)
        return len(tokens) < 80

    print "Total labels : " + str(len(labels))
    labels = filter(is_short, labels)
    print "After filtering long labels: " + str(len(labels))
    data_set_x = numpy.array([extract_x(label) for label in labels])
    data_set_y = [label2hotv(label) for label in labels]

    return DataSet(data_set_x, data_set_y, idx2vect, word2idx, len_embedding)

def load_data_split_sa(label_path, path_article, w2v_path, len_embedding):
    word2idx, idx2vect, labels, article_dic = load_data_common(label_path, len_embedding, path_article, w2v_path)

    def extract_x(label):
        article = article_dic[(label[IDX_R_ARTICLE], label[IDX_R_THREAD])]
        content = get_content(article)
        sentences = split_sentence(content)

        def contain_keyword(str):
            return label[IDX_R_KEYWORD] in str
        focus_sentences = list(filter(contain_keyword, sentences))
        tokens = flatten(tokenize_list(focus_sentences))
        return convert2index(tokens, word2idx)

    print "Total labels : " + str(len(labels))
    data_set_x = numpy.array([extract_x(label) for label in labels])
    data_set_y = [label2hotv(label) for label in labels]

    return DataSet(data_set_x, data_set_y, idx2vect, word2idx, len_embedding)

def load_data_ee(label_path, path_article, w2v_path, len_embedding):

    word2idx, idx2vect, labels, article_dic = load_data_common(label_path, len_embedding, path_article, w2v_path)

    all_keyword = set([label[IDX_R_KEYWORD] for label in labels])
    keyword2idx = set2idxmap(all_keyword)

    def extract_x(label):
        article = article_dic[(label[IDX_R_ARTICLE], label[IDX_R_THREAD])]
        content = get_content(article)
        tokens = tokenize(content)
        return convert2index(tokens, word2idx)

    print "Total labels : " + str(len(labels))
    data_set_x = numpy.array([extract_x(label) for label in labels])
    data_set_e = numpy.array([keyword2idx[label[IDX_R_KEYWORD]] for label in labels])
    data_set_y = [label2hotv(label) for label in labels]

    return EMDataSet(data_set_x, data_set_e, data_set_y, idx2vect, word2idx, len_embedding, keyword2idx)


def load_data_split_ee(label_path, path_article, w2v_path, len_embedding):

    word2idx, idx2vect, labels, article_dic = load_data_common(label_path, len_embedding, path_article, w2v_path)

    all_keyword = set([label[IDX_R_KEYWORD] for label in labels])
    keyword2idx = set2idxmap(all_keyword)

    def extract_x(label):
        article = article_dic[(label[IDX_R_ARTICLE], label[IDX_R_THREAD])]
        content = get_content(article)
        sentences = split_sentence(content)

        def contain_keyword(str):
            return label[IDX_R_KEYWORD] in str

        focus_sentences = list(filter(contain_keyword, sentences))
        tokens = flatten(tokenize_list(focus_sentences))
        return convert2index(tokens, word2idx)


    print "Total labels : " + str(len(labels))
    data_set_x = numpy.array([extract_x(label) for label in labels])
    data_set_e = numpy.array([keyword2idx[label[IDX_R_KEYWORD]] for label in labels])
    data_set_y = [label2hotv(label) for label in labels]

    return EMDataSet(data_set_x, data_set_e, data_set_y, idx2vect, word2idx, len_embedding, keyword2idx)



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
    return DataSet_validate(dataset_x, dataset_y, idx2vect, dimension)


def load_data_gt(pos_path, neg_path, w2v_path, dimension):
    print("Loading Data...")
    raw_pos = codecs.open(pos_path, "r", "cp949").readlines()
    raw_neg = codecs.open(neg_path, "r", "cp949").readlines()
    raw_data = raw_pos+raw_neg
    token_data = tokenize_list(raw_data)
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
    return EMDataSet(dataset_x, dataset_y, idx2vect, dimension)


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
    return EMDataSet(dataset_x, dataset_y, idx2vect, dimension)


entity_embed_length = 50


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

    data = load_data_ee(path_lable, path_article, w2v_path, len_embedding)

    #model = get_model_dirty(len_embedding, data, [3,4,5,6,7])
    model = get_entity_masking_model(len_embedding, data, [3, 4, 5, 6, 7])

    print("Training Model...")
    model.fit([data.train_x, data.train_e], data.train_y,
              batch_size=300,
              nb_epoch=300,
              validation_data=([data.test_x, data.test_e], data.test_y),
              shuffle=True)

    return model


def show_data(data):
    for i in range(30):
        print "x:",
        print data.x[i]
        print "y:",
        print data.y[i]

def run_sa():
    len_embedding = 100

    load_pickle = config_load_pickle_for_data
    if load_pickle:
        data = pickle.load(open(config_pickle_load_path,"rb"))
        print("Loading data from pickle")
    else:
        path_lable = "..\\input\\unanimous_label.csv"
        w2v_path = "..\\input\\korean_word2vec_wv_100.txt"
        path_article = "..\\input\\bobae_car_tkn_twitter.csv"
        data = load_data_sa(path_lable, path_article, w2v_path, len_embedding)
        pickle.dump(data, open(config_pickle_save_path, "wb"))

    model = get_simple_model(len_embedding, data, [3,4,5])
    init_weight = model.get_weights()

    X = data.x
    y = data.y


    accuracy_list = []

    for train_idx, valid_idx in StratifiedKFold(n_splits=config_n_fold).split(X, numpy.argmax(y, 1)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        print("Training Model...")
        model.set_weights(init_weight)
        hist = model.fit(X_train, y_train,
                  batch_size = config_batch_size,
                  nb_epoch = config_epoch,
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
        print("\nAccuracy : %f (at %d)\n" % (max(hist.history['val_acc']), numpy.argmax(hist.history['val_acc'])) )
        accuracy_list.append(max(hist.history['val_acc']))

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    std = numpy.std(accuracy_list)
    print("\n-Average accuracy : %f (%f)" % (avg_accuracy, std))

    return model, data.idx2vect, data.word2idx


def run_ee_sa():
    len_embedding = 100

    load_pickle = config_load_pickle_for_data
    if load_pickle:
        data = pickle.load(open(config_pickle_load_path,"rb"))
        print("Loading data from pickle")
    else:
        #path_lable = "..\\input\\corpus_samba.csv"
        path_lable = "..\\input\\unanimous_label.csv"
        w2v_path = "..\\input\\korean_word2vec_wv_100.txt"
        path_article = "..\\input\\bobae_car_tkn_twitter.csv"
        data = load_data_ee(path_lable, path_article, w2v_path, len_embedding)
        pickle.dump(data, open(config_pickle_save_path, "wb"))

    model = get_entity_masking_model(len_embedding, data, [3])
    init_weight = model.get_weights()

    X = data.x
    e = data.e
    y = data.y


    accuracy_list = []

    for train_idx, valid_idx in StratifiedKFold(n_splits=config_n_fold).split(X, numpy.argmax(y, 1)):
        X_train, e_train, y_train = X[train_idx], e[train_idx], y[train_idx]
        X_valid, e_valid, y_valid = X[valid_idx], e[valid_idx], y[valid_idx]
        print("Training Model...")
        model.set_weights(init_weight)
        hist = model.fit([X_train, e_train], y_train,
                  batch_size = config_batch_size,
                  nb_epoch = config_epoch,
                  validation_data=([X_valid, e_valid], y_valid),
                  shuffle=True)
        print("\nAccuracy : %f (at %d)\n" % (max(hist.history['val_acc']), numpy.argmax(hist.history['val_acc'])) )
        accuracy_list.append(max(hist.history['val_acc']))

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    std = numpy.std(accuracy_list)
    print("\n-Average accuracy : %f (%f)" % (avg_accuracy, std))

    return model, data.idx2vect, data.word2idx


def train_ee():
    len_embedding = 100

    load_pickle = config_load_pickle_for_data
    if load_pickle:
        data = pickle.load(open(config_pickle_load_path,"rb"))
        print("Loading data from pickle")
    else:
        #path_lable = "..\\input\\corpus_samba.csv"
        path_lable = "..\\input\\unanimous_label.csv"
        w2v_path = "..\\input\\korean_word2vec_wv_100.txt"
        path_article = "..\\input\\bobae_car_tkn_twitter.csv"
        data = load_data_ee(path_lable, path_article, w2v_path, len_embedding)
        pickle.dump(data, open(config_pickle_save_path, "wb"))

    #model = get_entity_masking_model(len_embedding, data, [3])
    model = entityMatchingRnn(len_embedding, len_sentence, data)
    init_weight = model.get_weights()

    X = data.x
    e = data.e
    y = data.y



    for train_idx, valid_idx in StratifiedKFold(n_splits=config_n_fold).split(X, numpy.argmax(y, 1)):
        X_train, e_train, y_train = X[train_idx], e[train_idx], y[train_idx]
        X_valid, e_valid, y_valid = X[valid_idx], e[valid_idx], y[valid_idx]
        print("Training Model...")
        model.set_weights(init_weight)
        hist = model.fit([X_train, e_train], y_train,
                  batch_size = config_batch_size,
                  nb_epoch = config_epoch,
                  validation_data=([X_valid, e_valid], y_valid),
                  shuffle=True)
        print("\nAccuracy : %f (at %d)\n" % (max(hist.history['val_acc']), numpy.argmax(hist.history['val_acc'])) )
        break

    return model, data.idx2vect, data.word2idx, data.keyword2idx


import pickle
def save_word2idx():
    len_embedding = 50

    path_lable = "..\\input\\corpus_samba.csv"
    w2v_path = "..\\input\\korean_word2vec_wv_50.txt"
    path_article = "..\\input\\bobae_car_tkn_twitter.csv"
    word2idx = load_data_get_word2idx(path_lable, path_article, w2v_path, len_embedding)

    pickle.dump(word2idx, open("word2idx","wb"))

def test_model(modeled):
    model = modeled[0]
    idx2vect = modeled[1]
    word2idx = modeled[2]
    keyword2idx = modeled[3]


    def predict(model, sentence, entity):
        tokens = tokenize(sentence)
        e_sentence = convert2index(tokens, word2idx)
        e_entity = keyword2idx[entity]
        X = numpy.array([e_sentence, e_entity])
        print model.predict(X)

    test_sentence = raw_input()
    while test_sentence:
        predict(model, test_sentence, "ignored")
        test_sentence = raw_input()

if __name__ == "__main__":
    #run_agreement()
    #run_carsurve()
    #modeled = run_sa()


    #run_ee_sa()
    #run_split_sa()
    modeled = train_ee()
    test_model(modeled)

    play_process_completed()