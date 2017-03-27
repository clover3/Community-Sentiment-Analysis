import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from idx2word import Idx2Word
from model import MemN2N
from eadata import TestCase  ## required for pickle

flags = tf.app.flags

flags.DEFINE_integer("n_entity", 240, "number of entity")
flags.DEFINE_integer("max_entity", 10, "maximal number of entity per sentence")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")

flags.DEFINE_integer("sdim", 99, "word embedding size")
flags.DEFINE_integer("edim", 55, "entity embedding size")

flags.DEFINE_integer("sent_len", 100, "maximum number of token in sentence")
flags.DEFINE_integer("mem_size", 48, "memory size [100]")

flags.DEFINE_float("init_lr", 1000., "initial learning rate [0.01]")

flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

flags.DEFINE_string("checkpoint_dir", "checkpoint_dir", "checkpoint_dir")
flags.DEFINE_string("data_name", "small", "data set name [ptb]")


import pickle

if "__main__" == __name__ :
    idx2word = Idx2Word("data\\idx2word")
    train_data = pickle.load(open("data\\dataSet1.p","rb"))
    valid_data = pickle.load(open("data\\dataSet2.p", "rb"))

    flags.DEFINE_integer("nwords", idx2word.voca_size, "number of words in corpus")
    with tf.Session() as sess:
        model = MemN2N(flags.FLAGS, sess)
        model.build_model()

        #if flags.FLAGS.is_test:
        #    model.run(valid_data, test_data)
        #else:
        model.run(train_data, valid_data)