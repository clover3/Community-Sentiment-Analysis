import os
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange
from view import Logger
import time
from clover_lib import *
import pickle
from itertools import groupby


def split_train_test(all_data, n_fold=3):
    def get_thread(test_case):
        return test_case.thread_id

    groups = []
    for k, g in groupby(all_data, get_thread):
        groups.append(list(g))
    random.shuffle(groups)

    folds = []
    for i in range(n_fold):
        folds.append([])

    idx = 0
    for group in groups:
        folds[idx].append(group)
        idx = (idx + 1) % n_fold

    r = []
    for test_idx in range(n_fold):
        train = flatten(flatten(folds[0:test_idx] + folds[test_idx+1:]))
        test  = flatten(folds[test_idx])
        print("test_idx = {} train len = {} test len = {}".format(test_idx, len(train), len(test)))
        r.append((train, test))

    return r


def base_accuracy(test_case):
    counter = FailCounter()
    for case in test_case:
        what_i_see = set(case.explicit_entity[-1])
        what_really_is = set(case.real_entity)

        if what_i_see == what_really_is :
            counter.suc()
        else:
            counter.fail()

    return counter.precision()


def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)

def to_unit_vector(tensor):
    sum = tf.reduce_sum(tensor, 1)
    return tensor / sum


def assert_shape(tensor, shape):
    if tensor.shape != shape:
        print_shape("Tensor shape error - ", tensor)
    assert (tensor.shape == shape)


def load_vec(file_name, idx2word, binary = True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("Loading word2vec...")
    w2v_cache = "cache\\w2v"
    if os.path.isfile(w2v_cache):
        return pickle.load(open(w2v_cache,"rb"))
    vocab = idx2word.get_voca()

    mode = ("rb" if binary else "r")
    word_vecs = {}
    ndim = 0
    with open(file_name, mode, encoding="utf-8") as f:
        header = f.readline()
        vocab_size, ndim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * ndim

        def getline():
            if binary:
                return np.fromstring(f.read(binary_len), dtype='float32')
            else:
                return np.array(f.readline().split(), dtype='float32')

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
                idx = idx2word.word2idx(word)
                word_vecs[idx] = getline()
            else:
                getline()

    l = []
    for i in range(idx2word.voca_size):
        if i in word_vecs:
            l.append(word_vecs[i])
        else:
            l.append(np.random.uniform(-0.25, 0.25, ndim))

    r = np.ndarray(shape=[idx2word.voca_size, ndim], buffer= np.array(l))
    print("Loaded word2vec...")
    pickle.dump(r, open(w2v_cache, "wb"))
    return r

class MemN2N(object):
    def __init__(self, config, sess, w2v = None):
        self.nwords = config.nwords            ## voca size
        self.n_entity = config.n_entity
        self.max_entity = config.max_entity
        self.init_std = config.init_std        ## std to initialize random
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.optimizer = config.optimizer
        #### only 1 hop
        self.sdim = config.sdim               ## sentence expression dimension
        self.edim = config.edim                ## entity expression dimension

        self.sent_len = config.sent_len        ## max sentence length
        self.mem_size = config.mem_size
        ### only linear
        ###self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir
        self.train_target = config.train_target

        self.content = tf.placeholder(tf.int32, [None, self.mem_size, self.sent_len], name="content")
        self.explicit_entity = tf.placeholder(tf.int32, [None, self.mem_size, self.max_entity], name="explicit_entity")
        self.real_entity = tf.placeholder(tf.int32, [None, self.max_entity,], name="real_entity")
        self.target_idx = tf.placeholder(tf.int32, [None, 1], name="target_idx")
        self.candidate_entity = tf.placeholder(tf.int32, [None, self.max_entity*2], name="candidate_entity")

        self.LE = None
        self.DE = None

        self.SE = None
        self.QE = None

        self.EM = None

        self.EE = None
        self.SEE = None

        self.W = None
        self.b = None

        self.memory_content = []
        self.memory_entity = []
        self.memory_location = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []
        self.last_weight = []

        self.logger = Logger()
        self.accuracy = dict()
        self.update_op_acc = dict()

        ## Debugging variable
        self.label = None
        self.prediction = None
        self.match = None

    def init_LE(self):
        option = 'random'
        option = 'fixed'
        if option=='fixed':
            base = [0.5] * 3 + [0] * (self.mem_size-3)
            tensor_1d = tf.constant(base, dtype=tf.float32)
            tensor = tf.reshape(tensor_1d, [self.mem_size, 1])
            embedding = tf.Variable(tensor, name="LE")
            return embedding
        else:
            return tf.Variable(tf.random_normal([self.mem_size, 1], stddev=self.init_std))

    def init_DE(self):
        option = 'random'
        option = 'fixed'
        if option=='fixed':
            def formula(i):
                peak = 0.5
                val = peak * ( 1 - i * 0.1 )
                if val < 0 :
                    return 0
                else:
                    return val
            base = [formula(i) for i in range(self.mem_size)]
            tensor_1d = tf.constant(base, dtype=tf.float32)
            tensor = tf.reshape(tensor_1d, [self.mem_size, 1])
            embedding = tf.Variable(tensor, name="DE")
            return embedding
        else:
            return tf.Variable(tf.random_normal([self.mem_size, 1], stddev=self.init_std))

    def init_EE(self, size):
        option = "identity"
        if option == 'random':
            return tf.Variable(tf.random_normal([self.n_entity, self.edim], stddev=self.init_std))
        else:
            iden = np.identity(size)
            first_row = np.zeros([1, size])
            ee_mat = np.concatenate((first_row, iden), 0)
            return tf.Variable(initial_value=ee_mat, dtype=tf.float32, trainable=False)

    def activate_label(self, prob_tensor):
        bias_value = -0.5
        bias = tf.constant(bias_value, dtype=tf.float32, shape=prob_tensor.shape)
        return tf.round(tf.sigmoid(tf.add(prob_tensor, bias)))

    def all_match(self, tensor1, tensor2):
        assert_shape(tensor1, (self.batch_size, self.edim))
        assert_shape(tensor2, (self.batch_size, self.edim))

        tensor_diff_boolean = tf.not_equal(tensor1, tensor2) ## cell will have 1 if not equal
        assert_shape(tensor_diff_boolean, (self.batch_size, self.edim))
        tensor_diff = tf.cast(tensor_diff_boolean, tf.float32)

        wrong_array = tf.reduce_sum(tensor_diff, 1)  ## any 1 in the batch means fail
        assert_shape(wrong_array, (self.batch_size,)) ## array of zero means perfect
        return wrong_array

    def update_accuracy(self, m_result, real_m, name):
        labels = self.activate_label(real_m)
        prediction = self.activate_label(m_result)
        self.label = labels
        self.prediction = prediction
        wrong_indicator = self.all_match(labels, prediction)
        self.match = wrong_indicator
        zeros = tf.zeros([self.batch_size], dtype=tf.int32)
        return tf.contrib.metrics.streaming_accuracy(wrong_indicator, zeros, name=name)

    def init_embedding(self):
        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.QE = tf.Variable(tf.random_normal([self.sdim * 2, 1], stddev=self.init_std))

        self.EM = self.init_EE(self.edim)


        self.LE = self.init_LE()
        self.DE = self.init_DE()

        self.EE = tf.Variable(tf.random_normal([self.n_entity, self.sdim, 1], stddev=self.init_std), name="EE")
        self.SEE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std), name="SEE", dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1]), name="b")

        self.W = tf.Variable(tf.constant(0.3, shape=[3, 1]), name="W")
        self.W4 = tf.Variable(tf.constant([0.0], shape=[1]))


    def build_model(self, run_names):
        print("Building Model...", end="")
        self.global_step = tf.Variable(0, name="global_step")
        self.init_embedding()

        m_result = self.memory_network(self.content, self.explicit_entity, self.target_idx)
        real_m = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.EM, self.real_entity), 1), (self.batch_size, self.edim))
        self.real_m = real_m
        self.m_result = m_result
        self.loss = tf.losses.mean_squared_error(real_m, m_result)

        ## train / valid / test / etc
        for name in run_names:
            self.accuracy[name], self.update_op_acc[name] = self.update_accuracy(m_result, real_m, name)

        self.lr = tf.Variable(self.current_lr)
        if self.optimizer == 'Adagrad':
            print("Using AdagradOptimizer")
            self.optim = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer == 'Adam':
            print("Using AdamOptimizer")
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            print("Using GradientDescent")
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
            params = None
            if self.train_target == 1:
                params = [self.LE, self.DE, self.W]
                print("Param : LE, DE, W")
            elif self.train_target == 2:
                params = [self.LE, self.DE, self.SE, self.QE, self.W]
                print("Param : LE, DE, SE, QE, W")
            elif self.train_target == 3:
                params = [self.LE, self.DE, self.W, self.SE, self.QE, self.W4, self.b, self.SEE]
                print("Param : [LE, DE], [W], [SE, QE], [b, SEE, W4]")
            else:
                print(self.train_target)

            grads_and_vars = self.opt.compute_gradients(self.loss, params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]

            inc = self.global_step.assign_add(1)
            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        self.W_out = self.W
        self.saver = tf.train.Saver({"LE":self.LE, "DE":self.DE, "W":self.W, "SE":self.SE, "QE":self.QE})
        tf.global_variables_initializer().run()
        print("Complete Building Model")

    def probability_and(self, tensor, axis):
        return tf.reduce_sum(tensor, axis)
        one = tf.ones_like(tensor)
        cap = tf.minimum(tensor, one)
        nots = one - cap

        not_prob = tf.reduce_prod(nots, axis)
        ones = tf.ones_like(not_prob)
        return ones - not_prob

    def make_memory_empty(self):
        # Step 0
        self.memory_content = []
        self.memory_entity = []
        self.memory_location = []  # [context_len * batch]

    def memory_network(self, content, explicit_entity, target_idx):
        self.make_memory_empty()

        mi_ = tf.nn.embedding_lookup(self.EM, explicit_entity[:, 0])  ## [batch * edim]
        m0 = tf.reduce_sum(mi_, 1)
        self.memory_content.append(content[:,0])
        self.memory_entity.append(m0)
        self.memory_location.append(tf.zeros([self.batch_size,], tf.int32))
        ## TODO : too many iteration...
        for context_len in range(1, self.mem_size):
            # 1) Select relevant article

            # 1-1) Location -----------
            weights_1 = tf.transpose(tf.nn.embedding_lookup(self.LE, self.memory_location), [1,0,2])  ## [batch * context_len * 1]


            # 1-2) Distance ------------
            dist = [context_len - x for x in self.memory_location]
            weights_2 = tf.transpose(tf.nn.embedding_lookup(self.DE, dist), [1,0,2])  ## [batch * context_len * 1]

            # 1-3) Sentence - Sentence --------
            memory_text = tf.transpose(tf.nn.embedding_lookup(self.SE, self.memory_content), [1,0,2,3]) ## [batch * context_len * sent_len * sdim ]
            memory_text_BoW = tf.reduce_sum(memory_text, 2)  ## [batch * context_len * sdim]

            query_text = tf.nn.embedding_lookup(self.SE, content[:, context_len])  # [batch * sent_len * sdim]
            query_text_BoW = tf.reduce_sum(query_text, 1)  ## [batch * sdim]
            query_text_raw_tile = tf.reshape(tf.tile(query_text_BoW, [context_len, 1]), [context_len, self.batch_size, self.sdim])
            query_text = tf.transpose(query_text_raw_tile, [1,0,2])
            assert(query_text.shape == (self.batch_size,context_len, self.sdim) )

            sent_feature = tf.concat([memory_text_BoW, query_text], 2)  # [ batch * context_len * (sdim*2)]
            QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])

            weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]
            assert_shape(weights_3, (self.batch_size, context_len, 1))

            # Weigt Sum
            weight_concat = tf.concat([weights_1, weights_2, weights_3], 2) # [batch * context_len * 2]
            assert_shape(weight_concat, (self.batch_size, context_len, 3))
            W_tile = tf.tile(self.W, [self.batch_size, 1])
            W_tile = tf.reshape(W_tile, [self.batch_size, 3, 1])  #[batch * 2 * 1]

            base_weights = tf.matmul(weight_concat, W_tile)  # [batch * context_len * 1]

            Cx = tf.transpose(tf.stack(self.memory_entity), [1, 2, 0])  ## [batch * edim * context_len]
            # 1-Final) Sum up
            raw_weights = tf.scalar_mul(1./context_len, base_weights) # [batch * context_len * 1]

            m = tf.reshape(tf.matmul(Cx, raw_weights), [self.batch_size, self.edim])  ## [batch * edim]
            self.last_weight = raw_weights
            # 2) Updated sum of entity
            mi_ = tf.nn.embedding_lookup(self.EM, explicit_entity[:, context_len])  ## [batch * max_entity * edim]
            m0 = tf.reduce_sum(mi_, 1) # [batch * dim]

            mp = m0 + m
            ones = tf.ones([self.batch_size, self.edim,])
            mp = tf.minimum(mp, ones)

            # 3) Update
            self.memory_content.append(content[:,context_len])
            self.memory_entity.append(mp)
            self.memory_location.append(tf.constant(context_len, tf.int32, [self.batch_size,]))

        all_entity = tf.transpose(self.memory_entity, [1,0,2])  # [batch * context * edim]

        result = []
        for i in range(self.batch_size):
            idx = target_idx[i][0]
            entity_i = all_entity[i][idx]
            self.ex_e = explicit_entity[i][idx]
            result.append(entity_i)
        m_result = tf.reshape(result, [self.batch_size, self.edim])
        return m_result

    def feature_location(self):
        weights_1 = tf.transpose(tf.nn.embedding_lookup(self.LE, self.memory_location), [1, 0, 2])
        # [batch * context_len * 1]
        return weights_1

    def feature_distance(self, j):
        j_2d = tf.constant(value=j, shape=[self.batch_size, ], dtype=tf.int32)
        dist = [tf.subtract(j_2d, i) for i in self.memory_location]
        weights_2 = tf.transpose(tf.nn.embedding_lookup(self.DE, dist), [1, 0, 2])  # [batch * context_len * 1]
        # weights_2 = tf.zeros([self.batch_size, j, 1])
        return weights_2

    def feature_sentence_similarity(self, content, j):

        Ax_i = tf.nn.embedding_lookup(self.SE, self.memory_content)  ## [context_len * batch * sent_len * sdim ]
        mi = tf.reduce_sum(Ax_i, 2)  ## [batch * context_len * sdim]

        Bx_j = tf.nn.embedding_lookup(self.SE, content[:, j])  # [batch * sent_len * sdim]
        bag_of_words = tf.reduce_sum(Bx_j, 1)  ## [batch * sdim]
        u_bar = tf.reshape(tf.tile(bag_of_words, [j, 1]), [j, self.batch_size, self.sdim])

        sent_feature = tf.transpose((tf.concat([mi, u_bar], 2)), [1, 0, 2])  # [ batch * context_len * (sdim*2)]
        QE_bar = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])

        weights_3 = tf.matmul(sent_feature, QE_bar) # [ batch * context_len ]
        return weights_3

    def feature_sentence_entity(self, content, j):
        Ex_r = tf.nn.embedding_lookup(self.EE, self.candidate_entity)
        Ex = tf.reshape(Ex_r, shape=[self.batch_size, self.max_entity * 2, self.sdim])  ## [batch * max_entity*2 * sdim]
        query_r = tf.reduce_sum(tf.nn.embedding_lookup(self.SEE, content[:, j]), 1)
        Query = tf.reshape(query_r, shape=[self.batch_size, self.sdim, 1])  ## [batch * sdim * 1]
        affinity = tf.add(tf.matmul(Ex, Query), self.b)
        affinity_T = tf.transpose(affinity, [0, 2, 1])  ## [batch * 1 * max_entity*2]
        me_base = tf.nn.embedding_lookup(self.EM, self.candidate_entity)  # [batch * max_entity*2 * edim ]
        me = tf.reshape(tf.matmul(affinity_T, me_base), shape=[self.batch_size, self.edim]) * self.W4
        return me

    # if given data is larger than size, use tail with size
    def init_content(self, size, data, dim):
        arr = np.ndarray(size, dtype=np.int32)
        arr.fill(0)

        if dim == 2:
            for j, line in enumerate(data[-size[0]:]):
                for i, token in enumerate(line[-size[1]:]):
                    arr[j][i] = token
        elif dim == 1:
            for i, token in enumerate(data[-size[0]:]):
                arr[i] = token
        return arr


    def data2batch(self, data):
        batch_supplier = Batch(self.batch_size)
        for test_case in data:
            a = self.init_content([self.mem_size, self.sent_len], test_case.content, 2)
            b = self.init_content([self.mem_size, self.max_entity], test_case.explicit_entity, 2)
            c = self.init_content([self.max_entity], test_case.real_entity, 1)
            target_idx = np.ndarray([1, ], dtype=np.int32)
            idx = min([len(test_case.content)-1, self.mem_size-1])
            target_idx.fill(idx)
            d = target_idx
            mentioned_entity = list(set(flatten(test_case.explicit_entity)))
            e = self.init_content([self.max_entity*2], mentioned_entity, 1)
            single = (a, b, c, d, e)
            batch_supplier.enqueue(single)
        return batch_supplier

    def train(self, data):
        cost = 0
        n_test = len(data)
        debug = False

        batch_supplier = self.data2batch(data)
        begin = True
        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.content: batch[0],
                self.explicit_entity: batch[1],
                self.real_entity: batch[2],
                self.target_idx: batch[3],
                self.candidate_entity: batch[4]
            }
            (_, _,
             W, W4, LE, DE, m_result, real_m,
             label, prediction, match, last_weight,
            loss) = self.sess.run([self.optim, self.update_op_acc["train"],
                                   self.W, self.W4, self.LE, self.DE, self.m_result, self.real_m,
                                   self.label, self.prediction, self.match, self.last_weight,
                                               self.loss], feed_dict)
            if begin :
                begin = False
                self.logger.print("labels", label)
                self.logger.print("prediction", prediction)
                self.logger.print("match", match)
                self.logger.print("LE", LE)
                self.logger.print("DE", DE)
                self.logger.print("W", W)
                self.logger.print("m_result", m_result)
                self.logger.print("real_m", real_m)
                self.logger.print("weight", last_weight)
            cost += np.sum(loss)
        accuracy = self.sess.run([self.accuracy["train"]])
        return cost/n_test, accuracy

    def test(self, data):
        n_test = len(data)
        cost = 0

        batch_supplier = self.data2batch(data)

        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.content: batch[0],
                self.explicit_entity: batch[1],
                self.real_entity: batch[2],
                self.target_idx: batch[3],
                self.candidate_entity: batch[4]
            }
            (loss, _,
             ) = self.sess.run([self.loss, self.update_op_acc["test"],
                                       ], feed_dict)
            cost += np.sum(loss)
        accuracy = self.sess.run([self.accuracy["test"]])
        return cost/n_test, accuracy

    def load_weights(self, sent_embedding = None):

        ## 1 load DE,LE
#        self.saver.restore(self.sess, "checkpoint_dir\\MemN2N.model-0")

        ## 2. Init SE, QE
        if sent_embedding is not None:
            self.sess.run(self.SE.assign(sent_embedding))
 #           self.sess.run(self.QE.assign(sent_embedding))

        ## 3. Load SEE, EE
        #saver2 = tf.train.Saver({"SEE": self.SEE, "EE": self.EE})
        #saver2.restore(self.sess, "model\\SEnEE.model")

    def demo(self, model_name, data):
        self.saver.restore(self.sess, "checkpoint_dir\\" + model_name)

        def predict(data):
            batch_supplier = self.data2batch(data)
            result = []
            while batch_supplier.has_next():
                batch = batch_supplier.deque()
                feed_dict = {
                    self.content: batch[0],
                    self.explicit_entity: batch[1],
                    self.real_entity: batch[2],
                    self.target_idx: batch[3],
                    self.candidate_entity: batch[4]
                }
                (prediction,) = self.sess.run([self.prediction], feed_dict)
                ## prediciion : [self.batch_size, self.n_entity]
                (batch_size, n_entity) = prediction.shape

                for batch in range(batch_size):
                    entity = []
                    for i in range(n_entity):
                        if prediction[batch][i] > 0 :
                            entity.append(i+1)
                        elif prediction[batch][i] == 0 :
                            Nothing = 0
                        else:
                            raise Exception("Not expected")
                    result.append(entity)
            return result

        labels = predict(data)
        from idx2word import Idx2Word
        idx2word = Idx2Word("data\\idx2word")
        total = 0
        wrong = 0
        over_estimate = 0
        under_estimate = 0
        easy_correct = 0
        good_job = 0
        simple_case = 0
        for test_case, entity in zip(data, labels):

            idx = len(test_case.content) - 1
            real_n = len(test_case.real_entity)
            expl_n = len(test_case.explicit_entity[idx])
            pred_n = len(entity)

            if expl_n == real_n :
                simple_case += 1
            if real_n == pred_n == expl_n :
                easy_correct += 1

            if expl_n != real_n and real_n == pred_n:
                None

            if set(test_case.real_entity) != set(entity):
                print("------------")
                text = [idx2word.idx2word(item )for item in test_case.content[idx]]
                print(text)
                print("explicit / real / predicted : ")
                print(test_case.explicit_entity[idx])
                print(test_case.real_entity)
                print(entity)
                wrong += 1

            if real_n < pred_n :
                over_estimate += 1

            if real_n > pred_n :
                under_estimate += 1

            if real_n > expl_n and real_n == pred_n :
                good_job += 1

            total += 1

        print({ "total": total,
          "wrong":wrong,
          "simple_case" : simple_case,
          "good job" : good_job,
          "easy_correct" : easy_correct,
          "over esitmate": over_estimate,
          "under-estimate": under_estimate})


    def run(self, train_data, test_data):
        if not self.is_test:
            train_acc_last = 0
            for idx in xrange(self.nepoch):
                self.sess.run(tf.local_variables_initializer())

                test_loss, test_acc = self.test(test_data)

                start = time.time()
                train_loss, train_acc = self.train(train_data)
                acc_delta = train_acc[0]-train_acc_last
                train_acc_last = train_acc[0]
                elapsed = time.time() - start
                # Logging
                self.step, = self.sess.run([self.global_step])
                self.log_loss.append([train_loss, test_loss])
                #self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'train_perplexity': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'train_accuracy': train_acc,
                    'acc_delta' : acc_delta,
                    'valid_accuracy': test_acc,
                    'elapsed': int(elapsed)
                }
                print(state)
                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
#            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

