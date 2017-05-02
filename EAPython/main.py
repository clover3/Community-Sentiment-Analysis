import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from idx2word import Idx2Word
from model import MemN2N, MemN2N_LDA, MemN2N_LSTM, EntityInherit
from m1 import M1
from easolver import load_vec
from model import split_train_test, base_accuracy
from clover_lib import play_process_completed
from eadata import TestCase  ## required for pickle

flags = tf.app.flags

flags.DEFINE_integer("n_entity", 66, "number of entity")
flags.DEFINE_integer("edim", 66, "entity embedding size")

flags.DEFINE_integer("max_entity", 10, "maximal number of entity per sentence")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_integer("batch_size", 10, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 1000, "number of epoch to use during training [100]")

flags.DEFINE_integer("sdim", 100, "word embedding size")


flags.DEFINE_integer("sent_len", 101, "maximum number of token in sentence")
flags.DEFINE_integer("max_sent", 20, "memory size [20]")

flags.DEFINE_float("init_lr", 0.001, "initial learning rate [0.01]")

flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", True, "print progress [False]")

flags.DEFINE_string("checkpoint_dir", "checkpoint_dir", "checkpoint_dir")
flags.DEFINE_string("data_name", "small", "data set name [ptb]")
flags.DEFINE_string("optimizer", "Adam", "data set name [ptb]")

flags.DEFINE_integer("train_target", 2, "1 : {LE, DE, W}   2 : {All} ,  3: Temp")
flags.DEFINE_integer("use_small_entity", True, "")
flags.DEFINE_string("model_type", "BoW", "BoW, LDA, LSTM, INHERIT, M1")

import pickle
import random

if "__main__" == __name__ :
    random.seed(0)
    idx2word = Idx2Word("data\\idx2word")
    w2v = load_vec("data\\korean_word2vec_wv_100.txt", idx2word, False)
    flags.DEFINE_integer("nwords", idx2word.voca_size, "number of words in corpus")

    print("Loading data...", end="")
    if not flags.FLAGS.use_small_entity : # full entity
        train_data = pickle.load(open("data\\dataSet1.p","rb"))
        valid_data = pickle.load(open("data\\dataSet3.p", "rb"))
    elif flags.FLAGS.model_type == "LDA" :
        data = pickle.load(open("data\\dataSet_lda_s.p", "rb"))
        runs = split_train_test(data, 3)
    else :
        data = pickle.load(open("data\\dataSet4_s.p", "rb"))
        runs = split_train_test(data, 3)
    print("Done")

    summary = []
    for (train_data, valid_data) in runs:
        with tf.Session() as sess:
            print("Base accuracy [Train] : " + str(base_accuracy(train_data)))
            base_valid = base_accuracy(valid_data)
            print("Base accuracy [Valid] : " + str(base_valid))

            if flags.FLAGS.model_type == "LDA" :
                model = MemN2N_LDA(flags.FLAGS, sess)
            elif flags.FLAGS.model_type == "LSTM":
                print("Using LSTM")
                model = MemN2N_LSTM(flags.FLAGS, sess)
            elif flags.FLAGS.model_type == "INHERIT":
                print("Using INHERIT")
                model = EntityInherit(flags.FLAGS, sess)
            elif flags.FLAGS.model_type == "M1":
                print("Using M1")
                model = M1(flags.FLAGS, sess)
            else:
                print("Using MemN2N")
                model = MemN2N(flags.FLAGS, sess)

            print("Next build model")
            model.build_model(run_names=["train", "test"])
            if flags.FLAGS.is_test:
                print("Starting demo")
                model.demo("MemN2N.model-648", valid_data)
            else:
                model.load_weights(w2v)
                model.run(train_data, valid_data)
                model.print_report(base_valid)
                best_accuracy = model.get_best_valid('valid_accuracy')
                best_f = model.get_best_valid('valid_f')
                summary.append("{}\t{}\t{}".format(base_valid, best_accuracy, best_f))

    for s in summary:
        print(s)

    play_process_completed()