import pickle
import os
import time
from view import Logger
from model_common import *

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

        for line in range(vocab_size):
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

    counter = FailCounter()
    l = []
    for i in range(idx2word.voca_size):
        if i in word_vecs:
            l.append(word_vecs[i])
            counter.suc()
        elif i == 0:
            l.append(np.full(ndim, 0.0))
        else:
            counter.fail()
            l.append(np.random.uniform(-0.25, 0.25, ndim))

    print("{} of the words not found".format(1-counter.precision()))
    r = np.ndarray(shape=[idx2word.voca_size, ndim], buffer= np.array(l))
    print("Loaded word2vec...")
    pickle.dump(r, open(w2v_cache, "wb"))
    return r

class EASolver(object):
    def __init__(self, config, sess, w2v = None):

        self.nwords = config.nwords            ## voca size
        self.n_entity = config.n_entity
        self.max_entity = config.max_entity
        self.init_std = config.init_std        ## std to initialize random
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.optimizer = config.optimizer
        self.sdim = config.sdim               ## sentence expression dimension
        self.edim = config.edim                ## entity expression dimension
        self.max_sent = config.max_sent
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir
        self.train_target = config.train_target

        self.sent_len = config.sent_len  ## max sentence length

        self.content = tf.placeholder(tf.int32, [None, self.max_sent, self.sent_len], name="content")
        self.explicit_entity = tf.placeholder(tf.int32, [None, self.max_sent, self.max_entity], name="explicit_entity")
        self.real_entity = tf.placeholder(tf.int32, [None, self.max_entity,], name="real_entity")
        self.target_idx = tf.placeholder(tf.int32, [None, 1], name="target_idx")
        self.candidate_entity = tf.placeholder(tf.int32, [None, self.max_entity*2], name="candidate_entity")

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log = []
        self.last_weight = []

        self.logger = Logger()
        self.accuracy = dict()
        self.update_op_acc = dict()
        self.precision = dict()
        self.precision_op = dict()
        self.recall = dict()
        self.recall_op = dict()
        self.print_state = config.show

        self.global_step = None

        self.SE = None
        self.QE = None
        self.EM = None
        self.W = None

    def init_EM(self, size):
        iden = np.identity(size)
        first_row = np.zeros([1, size])
        ee_mat = np.concatenate((first_row, iden), 0)
        return tf.Variable(initial_value=ee_mat, dtype=tf.float32, trainable=False)

    def init_embedding(self):
        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.QE = tf.Variable(tf.random_normal([self.sdim * 2, 1], stddev=self.init_std))

        self.EM = self.init_EM(self.edim)
        self.W = tf.Variable(tf.constant(0.0, shape=[1, 1]), name="W")

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
        labels = activate_label(real_m)
        prediction = activate_label(m_result)
        self.label = labels
        self.prediction = prediction
        wrong_indicator = self.all_match(labels, prediction)
        self.match = wrong_indicator
        zeros = tf.zeros([self.batch_size], dtype=tf.int32)
        return tf.contrib.metrics.streaming_accuracy(wrong_indicator, zeros, name=name)


    def update_recall_precision(self, hidden_truth, hidden_prediction, name):
        labels = activate_label(hidden_truth)
        prediction = activate_label(hidden_prediction)
        precision, precision_update = tf.contrib.metrics.streaming_precision(prediction, labels, name=name)
        recall, recall_update = tf.contrib.metrics.streaming_recall(prediction, labels, name=name)
        return precision, precision_update, recall, recall_update


    def get_explicit_entity_at_target(self):
        explicit = []
        for i in range(self.batch_size):
            idx = self.target_idx[i][0]
            entity_i = self.explicit_entity[i][idx]
            explicit.append(entity_i)
        mi_ = tf.nn.embedding_lookup(self.EM, explicit)  ## [batch * max_entity * edim]
        m0 = tf.reduce_sum(mi_, 1)  # [batch * dim]
        return m0

    def hidden_prediction(self, predict_all):
        explicit = self.get_explicit_entity_at_target()
        return predict_all - explicit

    def hidden_truth(self):
        explicit = self.get_explicit_entity_at_target()
        real_all = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.EM, self.real_entity), 1),
                            (self.batch_size, self.edim))
        return real_all - explicit


    def build_model(self, run_names):
        print("[EASolver]Building Model...", end="")
        self.global_step = tf.Variable(0, name="global_step")
        self.init_embedding()

        m_result = self.network(self.content, self.explicit_entity, self.target_idx)
        real_m = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(self.EM, self.real_entity), 1), (self.batch_size, self.edim))
        self.real_m = real_m
        self.m_result = m_result
        self.loss = tf.losses.mean_squared_error(real_m, m_result)

        ## train / valid / test / etc
        for name in run_names:
            self.accuracy[name], self.update_op_acc[name] = self.update_accuracy(m_result, real_m, name)
            (self.precision[name], self.precision_op[name], self.recall[name], self.recall_op[name])\
                = self.update_recall_precision(self.hidden_truth(), self.hidden_prediction(m_result), name)

        inc = self.global_step.assign_add(1)
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
                params = [self.LE, self.DE, self.SE, self.W]
                print("Param : LE, DE, SE, QE, W")
            elif self.train_target == 3:
                params = [self.LE, self.DE, self.W, self.SE, self.QE, self.W4, self.b, self.SEE]
                print("Param : [LE, DE], [W], [SE, QE], [b, SEE, W4]")
            else:
                print(self.train_target)

            grads_and_vars = self.opt.compute_gradients(self.loss, params)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]

            with tf.control_dependencies([inc]):
                self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        self.saver = tf.train.Saver({"W":self.W, "SE":self.SE, "QE":self.QE})
        tf.global_variables_initializer().run()
        print("Complete Building Model")

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
                        if prediction[batch][i] > 0:
                            entity.append(i + 1)
                        elif prediction[batch][i] == 0:
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

            if expl_n == real_n:
                simple_case += 1
            if real_n == pred_n == expl_n:
                easy_correct += 1

            if expl_n != real_n and real_n == pred_n:
                None

            if set(test_case.real_entity) != set(entity):
                print("------------")
                text = [idx2word.idx2word(item) for item in test_case.content[idx]]
                print(text)
                print("explicit / real / predicted : ")
                print(test_case.explicit_entity[idx])
                print(test_case.real_entity)
                print(entity)
                wrong += 1

            if real_n < pred_n:
                over_estimate += 1

            if real_n > pred_n:
                under_estimate += 1

            if real_n > expl_n and real_n == pred_n:
                good_job += 1

            total += 1

        print({"total": total,
               "wrong": wrong,
               "simple_case": simple_case,
               "good job": good_job,
               "easy_correct": easy_correct,
               "over esitmate": over_estimate,
               "under-estimate": under_estimate})

    # if given data is larger than size, use tail with size
    def init_content(self, size, data, dim, type):
        arr = np.ndarray(size, dtype=type)
        arr.fill(0)

        if dim == 2:
            for j, line in enumerate(data[-size[0]:]):
                for i, token in enumerate(line[-size[1]:]):
                    arr[j][i] = token
        elif dim == 1:
            for i, token in enumerate(data[-size[0]:]):
                arr[i] = token
        return arr

    def init_sentence(self, test_case):
        return self.init_content([self.max_sent, self.sent_len], test_case.content, 2, np.int32)

    def load_weights(self, sent_embedding=None):

        ## 1 load DE,LE
        #        self.saver.restore(self.sess, "checkpoint_dir\\MemN2N.model-0")

        ## 2. Init SE, QE
        if sent_embedding is not None:
            self.sess.run(self.SE.assign(sent_embedding))
            #           self.sess.run(self.QE.assign(sent_embedding))

            ## 3. Load SEE, EE
            # saver2 = tf.train.Saver({"SEE": self.SEE, "EE": self.EE})
            # saver2.restore(self.sess, "model\\SEnEE.model")

    def data2batch(self, data):
        batch_supplier = Batch(self.batch_size)
        for test_case in data:
            a = self.init_sentence(test_case)
            b = self.init_content([self.max_sent, self.max_entity], test_case.explicit_entity, 2, np.int32)
            c = self.init_content([self.max_entity], test_case.real_entity, 1, np.int32)
            target_idx = np.ndarray([1, ], dtype=np.int32)
            idx = min([len(test_case.content)-1, self.max_sent-1])
            target_idx.fill(idx)
            d = target_idx
            mentioned_entity = list(set(flatten(test_case.explicit_entity)))
            e = self.init_content([self.max_entity*2], mentioned_entity, 1, np.int32)
            single = (a, b, c, d, e)
            batch_supplier.enqueue(single)
        return batch_supplier

    def train(self, data):
        cost = 0
        n_test = len(data)
        debug = False

        batch_supplier = self.data2batch(data)
        debug = True
        count = 0
        while batch_supplier.has_next():
            batch = batch_supplier.deque()
            feed_dict = {
                self.content: batch[0],
                self.explicit_entity: batch[1],
                self.real_entity: batch[2],
                self.target_idx: batch[3],
                self.candidate_entity: batch[4]
            }
            (_, _, _, _,
             W, m_result, real_m, QE,
            loss) = self.sess.run([self.optim, self.update_op_acc["train"], self.recall_op["train"], self.precision_op["train"],
                                   self.W, self.m_result, self.real_m, self.QE,
                                               self.loss], feed_dict)
            if debug :
                self.logger.print("QE", QE)
                self.logger.print("W", W)
                self.logger.print("m_result", m_result)
                self.logger.print("real_m", real_m)
            cost += np.sum(loss)
        accuracy, precision, recall = self.sess.run(
            [self.accuracy["train"],
             self.precision["train"],
             self.recall["train"]])
        return cost/n_test, accuracy, precision, recall

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
            (loss, _, _, _,
             ) = self.sess.run([self.loss, self.update_op_acc["test"], self.precision_op["test"], self.recall_op["test"]
                                       ], feed_dict)
            cost += np.sum(loss)
        accuracy, precision, recall = self.sess.run(
            [self.accuracy["test"],
             self.precision["test"],
             self.recall["test"]])
        return cost/n_test, accuracy, precision, recall

    def run(self, train_data, test_data):
        if not self.is_test:
            train_acc_last = 0
            best_v_acc = 0
            for idx in range(self.nepoch):
                self.sess.run(tf.local_variables_initializer())

                test_loss, test_acc, test_precision, test_recall = self.test(test_data)
                test_f  = 2*test_precision*test_recall/(test_precision+test_recall+0.001)

                self.logger.set_prefix("Epoch:{}\n".format(idx))
                start = time.time()
                train_loss, train_acc, train_precision, train_recall = self.train(train_data)
                acc_delta = train_acc - train_acc_last
                train_acc_last = train_acc
                elapsed = time.time() - start
                # Logging
                self.step, = self.sess.run([self.global_step])
                self.log_loss.append([train_loss, test_loss])

                state = {
                    'train_perplexity': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'train_accuracy': train_acc,
                    'acc_delta': acc_delta,
                    'valid_accuracy': test_acc,
                    'valid_f': test_f,
                    'elapsed': int(elapsed)
                }
                if self.print_state:
                    print(state)
                self.log.append(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx - 1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break
                if test_acc > best_v_acc:
                    best_v_acc = test_acc
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step=idx)
        else:
            #            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def print_report(self, base_valid):
        best_acc = self.get_best_valid('valid_accuracy')
        best_f = self.get_best_valid('valid_f')

        print("base: {} : best_acc(valid): {} best_f:{}".
              format(base_valid, best_acc, best_f))


    def get_best_valid(self, measure):
        max_valid = 0
        best_state = None
        for state in self.log:
            if state[measure] > max_valid:
                max_valid = state[measure]
                best_state = state
        return best_state[measure]
