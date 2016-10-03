from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Embedding, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adadelta
import cPickle
import numpy


seed = 7
numpy.random.seed(seed)

len_sentence = 64

def index_rize(text_list, k):
    voca = set()

    max_l = 0
    for text in text_list:
        tokens = text.split()
        if len(tokens) > max_l :
            max_l = len(tokens)
        voca.update(tokens)

    count = 1
    voca_dict = dict()
    for word in voca:
        voca_dict[word]= count
        count += 1

    max_l = len_sentence

    print "max_l" , max_l
    vectors = []
    for text in text_list:
        vector = list(map(lambda x: voca_dict[x], text.split()))
        while len(vector) < max_l :
            vector.append(0)
        vectors.append(numpy.array(vector, dtype='float32'))


    nv = numpy.array(vectors)
    return nv, len(voca_dict)

class DataSet:
    def __init__(self, train_x, train_y, test_x, test_y, voca_size, len_embedding):
        self.train_x = numpy.array(train_x)
        self.train_y = numpy.array(train_y)
        self.test_x = numpy.array(test_x)
        self.test_y = numpy.array(test_y)
        self.voca_size = voca_size
        self.len_embedding = len_embedding


def load_data(pos_path, neg_path, dimension):
    print("Loading Data...")
    raw_pos = open(pos_path).readlines()
    raw_neg = open(neg_path).readlines()

    dataset_x, voca_size = index_rize(raw_pos+raw_neg, dimension)
    dataset_y = len(raw_pos)*[1] + len(raw_neg) * [0]

    data = zip(dataset_x, dataset_y)
    numpy.random.shuffle(data)
    len_train = int(len(data) * 0.9)
    train_x, train_y = zip(*data[0:len_train])
    test_x, test_y = zip(*data[len_train:])

    data_set = DataSet(train_x, train_y, test_x, test_y, voca_size, dimension)
    return data_set


def get_model(len_embedding, size_voca):
    print("Creating model...")
    len_sentence = 100
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


def get_model_dirty(len_embedding, size_voca):
    print("Creating model...")

    dropout_prob = (0.5, 0.5)
    n_filter = 500
    hidden_dims = 100
    sent_input = Input(shape=(len_sentence,), dtype='int32', name='sent_level_input')
    sent_x = Embedding(size_voca + 1, len_embedding,
                       input_length=len_sentence)(sent_input)
    sent_x = Dropout(dropout_prob[0], input_shape=(len_sentence, len_embedding))(sent_x)

    multiple_filter_output= []
    convs = []
    pools = []
    filter_sizes = (3, 4, 5)
    for i in xrange(len(filter_sizes)):
        convs.append(Convolution1D(nb_filter=n_filter,
                         filter_length= filter_sizes[i],
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(sent_x))
        pools.append(MaxPooling1D(pool_length = len_sentence - filter_sizes[i] + 1)(convs[i]))
        multiple_filter_output.append(Flatten()(pools[i]))
    sent_v = merge(multiple_filter_output, mode = 'concat')

    sent_v = Dense(hidden_dims)(sent_v)
    sent_v = Dropout(dropout_prob[1])(sent_v)
    sent_v = Activation('relu')(sent_v)
    sent_loss = Dense(1, activation='sigmoid', name='sent_level_output')(sent_v)

    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

    model = Model(input=sent_input, output=sent_loss)
    #model.load_weights("YoonKim_mr.h5")
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adadelta)
    return model

def run():
    len_embedding = 300

    data = load_data("rt-polarity-small.pos", "rt-polarity-small.neg", len_embedding)

    model = get_model_dirty(len_embedding, data.voca_size)
    #model = get_model(len_embedding, data.voca_size)

    print("Training Model...")
    model.fit(data.train_x, data.train_y,
              batch_size=50,
              nb_epoch=10,
              validation_data=(data.test_x, data.test_y),
              shuffle=True)

run()