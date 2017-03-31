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
    sum = tf.reduce_sum(tensor, 1)
    return tensor / sum



def load_vec(file_name, idx2word, binary = True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("Loading word2vec...")
    #w2v_cache = "cache\\w2v"
    #if os.path.isfile(w2v_cache):
    #    return cPickle.load(open(w2v_cache,"rb"))
    vocab = idx2word.get_voca()

    mode = ("rb" if binary else "r")
    word_vecs = {}
    ndim = 0
    with open(file_name, mode) as f:
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

    embedding = np.ndarray()
    l = []
    for i in range(idx2word.voca_size):
        if i in word_vecs:
            l.append(word_vecs[i])
        else:
            l.append(np.random.uniform(-0.25,0.25,ndim))

    print("Loaded word2vec...")
#    cPickle.dump(word_vecs, open(w2v_cache, "wb"))
    return np.ndarray(l, dtype='float32')

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

        self.content = tf.placeholder(tf.int32, [None, self.mem_size, self.sent_len], name="content")
        self.explicit_entity = tf.placeholder(tf.int32, [None, self.mem_size, self.max_entity], name="explicit_entity")
        self.real_entity = tf.placeholder(tf.int32, [None, self.max_entity,], name="real_entity")
        self.target_idx = tf.placeholder(tf.int32, [None, 1], name="target_idx")

        self.SE = None
        self.QE = None
        self.EE = None
        self.LE = None
        self.DE = None
        self.W = None

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
        return tf.reduce_sum(tf.cast(tf.not_equal(tensor1, tensor2), tf.float32), 1)

    def update_accuracy(self, m_result, real_m):
        labels = self.activate_label(real_m)
        prediction = self.activate_label(m_result)
        d = self.all_match(labels, prediction)
        zeros = tf.zeros([self.batch_size], dtype=tf.int32)
        return tf.contrib.metrics.streaming_accuracy(d, zeros)



    def build_model(self):
        print("Building Model...", end="")
        self.global_step = tf.Variable(0, name="global_step")

        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.QE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.EE = self.init_EE(self.edim)
        self.LE = self.init_LE()
        self.DE = self.init_DE()
        self.W = tf.Variable(tf.constant(0.3, shape=[3, 1]))

        self.loss = 0
        m_result = self.memory_network(self.content, self.explicit_entity, self.target_idx)
        real_m = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.EE, self.real_entity), 1), m_result.shape)
        self.real_m = real_m
        self.m_result = m_result
        self.loss += tf.losses.mean_squared_error(real_m, m_result)

        self.accuracy, self.update_op_acc = self.update_accuracy(m_result, real_m)

        self.lr = tf.Variable(self.current_lr)
        if False:
            self.opt = tf.train.AdamOptimizer()
        else:
            self.opt = tf.train.GradientDescentOptimizer(self.lr)

            #params = [self.SE, self.QE, self.EE, self.LE, self.DE]
            params = [self.W, self.LE, self.DE]
            grads_and_vars = self.opt.compute_gradients(self.loss, params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]

            inc = self.global_step.assign_add(1)
            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        print("Complete Building Model")

    def memory_network(self, content, explicit_entity, target_idx):
        # Step 0
        self.memory_content = []
        self.memory_entity = []
        self.memory_location = []  # [context_len * batch]

        mi_ = tf.nn.embedding_lookup(self.EE, explicit_entity[:,0])  ## [batch * edim]
        m0 = tf.reduce_sum(mi_, 1)
        self.memory_content.append(content[:,0])
        self.memory_entity.append(m0)
        self.memory_location.append(tf.zeros([self.batch_size,], tf.int32))
        ## TODO : too many iteration...
        for j in range(1, self.mem_size):
            # 1) Select relevant article

            # 1-1) Location
            weights_1 = tf.transpose(tf.nn.embedding_lookup(self.LE, self.memory_location), [1,0,2])
            # [batch * context_len * 1]
            # 1-2) Distance

            # memory location :
            j_2d = tf.constant(value = j, shape= [self.batch_size,], dtype=tf.int32)
            dist = [tf.subtract(j_2d, i) for i in self.memory_location]
            weights_2 = tf.transpose(tf.nn.embedding_lookup(self.DE, dist), [1, 0, 2])  # [batch * context_len * 1]

            # 1-3) Sentence - Sentence
            Bx_j = tf.nn.embedding_lookup(self.QE, content[:,j])  # [batch * sent_len * sdim]
            u = tf.reduce_sum(Bx_j, 1)  ## [batch * sdim]

            Ax_i = tf.nn.embedding_lookup(self.SE, self.memory_content)  ## [context_len * batch * sent_len * sdim ]
            mi = tf.reduce_sum(Ax_i, 2)  ## [batch * context_len * sdim]
            mi_ = tf.transpose(mi, [1, 0, 2])
            u_2d = tf.reshape(u, [self.batch_size, self.sdim, -1]) ##
            weights_3 = tf.matmul(mi_, u_2d)  ## [batch * context_len]

            concat = tf.concat([weights_1, weights_2, weights_3], 2) # [batch * context_len * 2]
            W__ = tf.tile(self.W, [self.batch_size, 1])
            W_ = tf.reshape(W__, [self.batch_size, 3, 1])  #[batch * 2 * 1]

            raw_weights = tf.matmul(concat, W_)  # [batch * context_len * 1]
            # 1-Final) Sum up
            # weights = tf.add(weights_1, tf.add(weights_2, weights_3))
            weights = tf.scalar_mul(1./j, raw_weights) # [batch * context_len * 1]
            Cx = tf.transpose(tf.stack(self.memory_entity), [1,2,0])  ## [batch * edim * context_len]

            m = tf.reshape(tf.matmul(Cx, weights), [self.batch_size, self.edim])  ## [batch * edim * 1]

            # 2) Updated sum of entity
            mi_ = tf.nn.embedding_lookup(self.EE, explicit_entity[:,j])  ## [batch * max_entity * edim]
            m0 = tf.reduce_sum(mi_, 1)

            mp = tf.add(m, m0)
            ones = tf.ones([self.batch_size, self.edim,])
            mp = tf.minimum(mp, ones)

            # 3) Update
            self.memory_content.append(content[:,j])
            self.memory_entity.append(mp)
            self.memory_location.append(tf.constant(j, tf.int32, [self.batch_size,]))

        def index(tensor):
            nums = tf.reshape(tf.range(self.batch_size), [self.batch_size, 1])
            data = tf.reshape(tensor, [self.batch_size, 1])
            c = tf.concat([data, nums], 1)
            return tf.reshape(c, shape=[self.batch_size,2])

        # [context_len * batch * edim]
        source = tf.transpose(self.memory_entity, [1,0,2])  # [batch * context * edim]

        result = []
        for i in range(self.batch_size):
            idx = target_idx[i][0]
            piece = source[i][idx]
            result.append(piece)
        m_result = tf.reshape(result, [self.batch_size, self.edim])
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

    class Batch:
        def __init__(self, batch_size):
            self.data = []
            self.bach_size = batch_size
            self.index = 0

        def enqueue(self, sample):
            self.data.append(sample)

        def has_next(self):
            return len(self.data) >= self.index + self.bach_size

        def deque(self):
            num_input = len(self.data[0])
            input = [ [] for i in range(num_input)]
            for i in range(self.bach_size):
                for j in range(num_input):
                    item = self.data[self.index+i][j]
                    input[j].append(item)
            self.index += self.bach_size
            return input

    def data2batch(self, data):
        batch_supplier = self.Batch(self.batch_size)
        for test_case in data:
            a = self.init_content([self.mem_size, self.sent_len], test_case.content, 2)
            b = self.init_content([self.mem_size, self.max_entity], test_case.explicit_entity, 2)
            c = self.init_content([self.max_entity], test_case.real_entity, 1)
            target_idx = np.ndarray([1, ], dtype=np.int32)
            target_idx.fill(min([len(test_case.content), self.mem_size-1]))
            d = target_idx
            single = (a, b, c, d)
            batch_supplier.enqueue(single)
        return batch_supplier


    def train(self, data):
        cost = 0
        n_test = len(data)

        batch_supplier = self.data2batch(data)

        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.content: batch[0],
                self.explicit_entity: batch[1],
                self.real_entity: batch[2],
                self.target_idx: batch[3]
            }
            (_, _,
            loss) = self.sess.run([self.optim, self.update_op_acc,
                                               self.loss], feed_dict)
            cost += np.sum(loss)

        return cost/n_test

    def test(self, data, label):
        n_test = len(data)
        cost = 0

        batch_supplier = self.data2batch(data)

        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.content: batch[0],
                self.explicit_entity: batch[1],
                self.real_entity: batch[2],
                self.target_idx: batch[3]
            }
            (loss) = self.sess.run([self.loss], feed_dict)
            cost += np.sum(loss)

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
                accuracy, self.step = self.sess.run([self.accuracy, self.global_step])
                #self.log_loss.append([train_loss, test_loss])
                #self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

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

