import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from clover_lib import *
import tensorflow as tf
import numpy as np
import pickle
from model import load_vec, print_shape
from random import shuffle
from idx2word import Idx2Word


class S2E():
    def __init__(self, config, sess):
        self.nwords = config.nwords  ## voca size
        self.n_entity = config.n_entity
        self.init_std = config.init_std  ## std to initialize random
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.sdim = config.sdim
        self.edim = config.edim
        self.sent_len = config.sent_len        ## max sentence length
        self.max_grad_norm = config.max_grad_norm

        self.text = tf.placeholder(tf.int32, [None, self.sent_len], name="text")
        self.entity = tf.placeholder(tf.int32, [None, 1], name="entity")
        self.label = tf.placeholder(tf.int32, [None, 1], name="label")

        self.SE = None
        self.EE = None

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = 0
        self.step = None

        self.sess = sess

        self.accuracy = dict()
        self.acc_update = dict()

        self.train_op = None
        self.log_loss = []

    def build_model(self):
        print("Building Model...", end="")
        self.global_step = tf.Variable(0, name="global_step")

        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std), dtype=tf.float32)
        self.EE = tf.Variable(tf.random_normal([self.n_entity, self.sdim, 1], stddev=self.init_std), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1]))
        self.loss = 0

        v1 = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.SE, self.text), 1), shape=[self.batch_size, 1, self.sdim])  ## [batch * sdim]
        v2 = tf.reshape(tf.nn.embedding_lookup(self.EE, self.entity), shape=[self.batch_size, self.sdim, 1])  ## [ batch * sdim * 1]
        t = tf.matmul(v1,v2) + self.b
        y = tf.reshape(t, [self.batch_size, 1])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=y))
        self.loss += tf.losses.mean_squared_error(y, self.label)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.EE, self.b, self.SE]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                  for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.train_op = self.opt.apply_gradients(clipped_grads_and_vars)

        for name in ["test", "train"]:
            self.accuracy[name], self.acc_update[name] = tf.contrib.metrics.streaming_accuracy(tf.round(y), tf.round(self.label), name=name)
        tf.global_variables_initializer().run()
        print("Complete Building Model")

    def init_w2v(self, embedding):
        self.sess.run(self.SE.assign(embedding))

    def to_batch(self, data):
        batch_supplier = Batch(self.batch_size)

        def init_single(elem):
            arr = np.ndarray(1, dtype=np.int32)
            arr[0] = elem
            return arr

        for line in data:
            content = np.ndarray(self.sent_len, dtype=np.int32)
            content.fill(0)
            for i, token in enumerate(line[1][:self.sent_len]):
                content[i] = token
            a = init_single(line[0])
            b = content
            c = init_single(line[2])
            single = (a, b, c)
            batch_supplier.enqueue(single)
        return batch_supplier

    def train(self, train_data):
        cost = 0
        self.sess.run(tf.local_variables_initializer())

        batch_supplier = self.to_batch(train_data)
        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.entity: batch[0],
                self.text: batch[1],
                self.label: batch[2]
            }
            (_, _, loss) = self.sess.run([self.train_op, self.acc_update["train"], self.loss], feed_dict)
            cost += np.sum(loss)

        accuracy = self.sess.run([self.accuracy["train"]])
        return cost, accuracy

    def test(self, test_data):
        cost = 0
        self.sess.run(tf.local_variables_initializer())

        batch_supplier = self.to_batch(test_data)
        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.entity: batch[0],
                self.text: batch[1],
                self.label: batch[2]
            }
            (_, loss) = self.sess.run([self.acc_update["test"], self.loss], feed_dict)
            cost += np.sum(loss)

        accuracy = self.sess.run([self.accuracy["test"]])
        return cost, accuracy

    def run(self, data, w2v):
        self.build_model()
        self.init_w2v(w2v)

        shuffle(data)

        train_data = data[:-4000]
        test_data = data[-4000:]
        print("train data size : {}".format(len(train_data)))

        for idx in range(self.nepoch):
            self.sess.run(tf.local_variables_initializer())

            start = time.time()
            train_loss, train_acc = self.train(train_data)
            elapsed = time.time() - start

            test_loss, test_acc = self.test(test_data)

            state = {
                'train_perplexity': train_loss,
                'epoch': idx,
                'learning_rate': self.current_lr,
                'valid_perplexity': test_loss,
                'train_accuracy': train_acc,
                'valid_accuracy': test_acc,
                'elapsed': elapsed
            }
            print(state)
            if train_acc[0] > 0.6 :
                saver = tf.train.Saver({"SEE":self.SE, "EE":self.EE})
                saver.save(self.sess, "model\\SEnEE.model")
                break

            self.log_loss.append([train_loss, test_loss])

            if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx - 1][1] * 0.9999:
                self.current_lr = self.current_lr / 1.5
                self.lr.assign(self.current_lr).eval()
            if self.current_lr < 1e-5: break


flags = tf.app.flags

flags.DEFINE_integer("n_entity", 240, "number of entity")
flags.DEFINE_integer("edim", 50, "entity embedding size")

flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_integer("batch_size", 1000, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 10000, "number of epoch to use during training [100]")

flags.DEFINE_integer("sdim", 100, "word embedding size")
flags.DEFINE_integer("sent_len", 101, "maximum number of token in sentence")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")

flags.DEFINE_string("checkpoint_dir", "checkpoint_dir", "checkpoint_dir")


def S2E_train():
    print("Loading data...", end="")
    idx2word = Idx2Word("data\\idx2word")
    print("Done")
    w2v = load_vec("data\\korean_word2vec_wv_100.txt", idx2word, False)

    flags.DEFINE_integer("nwords", idx2word.voca_size, "number of words in corpus")
    with tf.Session() as sess:
        corpus = pickle.load(open("data\\E2S.p", "rb"))
        s2e = S2E(flags.FLAGS, sess)
        s2e.run(corpus, w2v)


if "__main__" == __name__ :
    S2E_train()