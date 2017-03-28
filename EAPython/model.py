import os
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange
from view import Logger
import time


def print_shape(text, matrix):
    print(text, end="")
    print(matrix.shape)

def to_unit_vector(tensor):
    sum = tf.reduce_sum(tensor, 0)
    return tensor / sum

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords            ## voca size
        self.n_entity = config.n_entity
        self.max_entity = config.max_entity
        self.init_std = config.init_std        ## std to initialize random
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
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

        self.content = tf.placeholder(tf.int32, [self.mem_size, self.sent_len], name="content")
        self.explicit_entity = tf.placeholder(tf.int32, [self.mem_size, self.max_entity], name="explicit_entity")
        self.real_entity = tf.placeholder(tf.int32, [self.max_entity,], name="real_entity")
        self.target_idx = tf.placeholder(tf.int32, [1], name="target_idx")

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
        self.accuracy = None
        self.update_op_acc = None

    def init_LE(self):
        option = 'random'
        option = 'fixed'
        if option=='fixed':
            base = [0.5] * 3 + [0] * (self.mem_size-3)
            tensor_1d = tf.constant(base, dtype=tf.float32)
            tensor = tf.reshape(tensor_1d, [self.mem_size, 1])
            embedding = tf.Variable(tensor)
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
            embedding = tf.Variable(tensor)
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
            return tf.Variable(initial_value=ee_mat, dtype=tf.float32)

    def activate_label(self, prob_tensor):
        bias_value = -0.5
        bias = tf.constant(bias_value, dtype=tf.float32, shape=prob_tensor.shape)
        return tf.round(tf.sigmoid(tf.add(prob_tensor, bias)))

    def all_match(self, tensor1, tensor2):
        return tf.reduce_sum(tf.cast(tf.not_equal(tensor1, tensor2), tf.float32), 0)


    def build_model(self):
        print("Building Model...", end="")
        self.global_step = tf.Variable(0, name="global_step")

        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.QE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.EE = self.init_EE(self.edim)
        self.LE = self.init_LE()
        self.DE = self.init_DE()
        self.W  = tf.Variable(tf.constant(0.3, shape=[2, 1]))

        # mp is predicted entity
        m_result = self.build_network()

        real_m = tf.reduce_sum(tf.nn.embedding_lookup(self.EE, self.real_entity), 0)
        self.real_m = real_m
        self.m_result = m_result

        labels = self.activate_label(real_m)
        prediction = self.activate_label(m_result)
        diff = self.all_match(labels, prediction)
        zeros = tf.zeros([1,], dtype=tf.int32)
        self.accuracy, self.update_op_acc = tf.contrib.metrics.streaming_accuracy(diff, zeros)
        self.loss = tf.losses.mean_squared_error(real_m, m_result)

        self.lr = tf.Variable(self.current_lr)
        if False:
            self.opt = tf.train.AdamOptimizer()
        else:
            self.opt = tf.train.GradientDescentOptimizer(self.lr)

            #params = [self.SE, self.QE, self.EE, self.LE, self.DE]
            params = [self.LE, self.DE, self.W]
            grads_and_vars = self.opt.compute_gradients(self.loss, params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]

            inc = self.global_step.assign_add(1)
            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        print("Complete Building Model")

    def build_network(self):
        # Step 0
        mi_ = tf.nn.embedding_lookup(self.EE, self.explicit_entity[0])  ## [edim]
        m0 = tf.reduce_sum(mi_, 0)
        self.memory_content.append(self.content[0])
        self.memory_entity.append(m0)
        self.memory_location.append(0)
        ## TODO : too many iteration...
        for j in range(1, self.mem_size):
            # 1) Select relevant article

            # 1-1) Location
            weights_1 = tf.nn.embedding_lookup(self.LE, self.memory_location)  ## [context_len * 1]

            # 1-2) Distance
            dist = [j - x for x in self.memory_location]
            weights_2 = tf.nn.embedding_lookup(self.DE, dist)  ## [context_len * 1]

            # 1-3) Sentence - Sentence
            Bx_j = tf.nn.embedding_lookup(self.QE, self.content[j])  ## [sent_len * sdim]
            u = tf.reduce_sum(Bx_j, 0)  ## [sdim]

            Ax_i = tf.nn.embedding_lookup(self.SE, self.memory_content)  ## [context_len * sent_len * sdim ]
            mi = tf.reduce_sum(Ax_i, 1)  ## [context_len * sdim]
            u_2d = tf.reshape(u, [self.sdim, -1])
            weights_3 = tf.matmul(mi, u_2d)  ## [context_len]

            concat = tf.concat([weights_1, weights_2], 1)
            raw_weights = tf.matmul(concat, self.W)
            # 1-Final) Sum up
            # weights = tf.add(weights_1, tf.add(weights_2, weights_3))
            weights = to_unit_vector(raw_weights)
            weights_T = tf.transpose(weights)
            Cx = tf.stack(self.memory_entity)  ## [context_len * edim]

            m = tf.reshape(tf.matmul(weights_T, Cx), [-1])  ## [edim]

            # 2) Updated sum of entity
            mi_ = tf.nn.embedding_lookup(self.EE, self.explicit_entity[j])  ## [edim]
            m0 = tf.reduce_sum(mi_, 0)

            mp = tf.add(m, m0)
            ones = tf.ones([self.edim,])
            mp = tf.minimum(mp, ones)


            # 3) Update
            self.memory_content.append(self.content[j])
            self.memory_entity.append(mp)
            self.memory_location.append(j)

        #mi_ = tf.nn.embedding_lookup(self.EE, self.explicit_entity[self.target_idx])  ## [edim]
        self.m0 = tf.reduce_sum(mi_, 0)

        m_result = tf.gather(self.memory_entity, self.target_idx)
        m_result = tf.reshape(m_result, [self.edim,])
        return m_result

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

    def train(self, data):
        cost = 0
        n_test = len(data)

        if self.show:
            from clover_lib import ProgressBar
            bar = ProgressBar('Train', max=n_test)

        for test_case in data:
            if self.show: bar.next()

            arr_content = self.init_content([self.mem_size, self.sent_len], test_case.content, 2)
            arr_ex_entity = self.init_content([self.mem_size, self.max_entity], test_case.explicit_entity, 2)
            arr_r_entity = self.init_content([self.max_entity], test_case.real_entity, 1)
            target_idx = np.ndarray([1, ], dtype=np.int32)
            target_idx.fill(len(test_case.content))

            (_, _,
             loss, self.step, real_e,
             m_result, memory_entity) = self.sess.run(
                [self.optim, self.update_op_acc,
                 self.loss, self.global_step, self.real_entity,
                 self.m_result, self.memory_entity
                 ], feed_dict = {
                                                    self.content: arr_content,
                                                    self.explicit_entity: arr_ex_entity,
                                                    self.real_entity: arr_r_entity,
                                                    self.target_idx : target_idx
                                                }
                                               )
            cost += np.sum(loss)
        if self.show: bar.finish()
        return cost/n_test

    def test(self, data, label):
        n_test = len(data)
        cost = 0

        if self.show:
            from clover_lib import ProgressBar
            bar = ProgressBar('Test', max=n_test)

        for test_case in data:
            if self.show: bar.next()

            arr_content = self.init_content([self.mem_size, self.sent_len], test_case.content, 2)
            arr_ex_entity = self.init_content([self.mem_size, self.max_entity], test_case.explicit_entity, 2)
            arr_r_entity = self.init_content([self.max_entity], test_case.real_entity, 1)
            target_idx = np.ndarray([1, ], dtype=np.int32)
            target_idx.fill(len(test_case.content))

            loss = self.sess.run([self.loss], feed_dict = {
                                                    self.content: arr_content,
                                                    self.explicit_entity: arr_ex_entity,
                                                    self.real_entity: arr_r_entity,
                                                    self.target_idx : target_idx
                                                })
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/n_test

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                self.sess.run(tf.local_variables_initializer())
                start = time.time()
                train_loss = np.sum(self.train(train_data))
                elapsed = time.time() - start
                test_loss = np.sum(self.test(test_data, label='Validation'))
                # Logging
                accuracy = self.sess.run([self.accuracy])
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'train_perplexity': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': test_loss,
                    'accuracy': accuracy,
                    'elapsed': elapsed
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

