import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from idx2word import Idx2Word
from model import MemN2N
from model import load_vec
from eadata import TestCase  ## required for pickle

flags = tf.app.flags

flags.DEFINE_integer("n_entity", 240, "number of entity")
flags.DEFINE_integer("edim", 240, "entity embedding size")

flags.DEFINE_integer("max_entity", 10, "maximal number of entity per sentence")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_integer("batch_size", 10, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")

flags.DEFINE_integer("sdim", 100, "word embedding size")


flags.DEFINE_integer("sent_len", 101, "maximum number of token in sentence")
flags.DEFINE_integer("mem_size", 50, "memory size [100]")

flags.DEFINE_float("init_lr", 0.1, "initial learning rate [0.01]")

flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", True, "print progress [False]")

flags.DEFINE_string("checkpoint_dir", "checkpoint_dir", "checkpoint_dir")
flags.DEFINE_string("data_name", "small", "data set name [ptb]")

flags.DEFINE_integer("train_target", 2, "1 : {LE, DE, W}   2 : {All} ,  3: Temp")

import pickle


if "__main__" == __name__ :
    print("Loading data...", end="")
    idx2word = Idx2Word("data\\idx2word")
    train_data = pickle.load(open("data\\dataSet1.p","rb"))
    valid_data = pickle.load(open("data\\dataSet3.p", "rb"))
    print("Done")
    w2v = load_vec("data\\korean_word2vec_wv_100.txt", idx2word, False)

    flags.DEFINE_integer("nwords", idx2word.voca_size, "number of words in corpus")
    with tf.Session() as sess:
        model = MemN2N(flags.FLAGS, sess)
        print("Next build model")
        model.build_model(run_names=["train", "test"])
        if flags.FLAGS.is_test:
            print("Starting demo")
            model.demo("MemN2N.model-648", valid_data)
        else:
            model.load_weights(w2v)
            model.run(train_data, valid_data)