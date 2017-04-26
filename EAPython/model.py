from easolver import EASolver
from clover_lib import *
from model_common import *

class MemN2N(EASolver):
    def __init__(self, config, sess, w2v = None):
        super(MemN2N, self).__init__(config, sess)

        self.mem_size = config.max_sent

        self.LE = None
        self.DE = None

        self.SE = None
        self.QE = None

        self.EE = None
        self.SEE = None

        self.W = None
        self.b = None

        self.memory_content = []
        self.memory_entity = []
        self.memory_location = []


        ## Debugging variable
        self.label = None
        self.prediction = None
        self.match = None

    def init_LE(self):
        option = 'random'
        option = 'fixed'
        if option=='fixed':
            predef = 6
            base = [0.0] * predef + [0] * (self.mem_size - predef)
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
                peak = 0.0
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

    def init_embedding(self):
        super(MemN2N, self).init_embedding()

        self.LE = self.init_LE()
        self.DE = self.init_DE()

        self.EE = tf.Variable(tf.random_normal([self.n_entity, self.sdim, 1], stddev=self.init_std), name="EE")
        self.SEE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std), name="SEE", dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1]), name="b")

        self.W = tf.Variable(tf.constant(0.00, shape=[3, 1]), name="W")
        self.W4 = tf.Variable(tf.constant([0.0], shape=[1]))

    def make_memory_empty(self):
        # Step 0
        self.memory_content = []
        self.memory_entity = []
        self.memory_location = []  # [context_len * batch]

    def init_first_slot(self, content, explicit_entity):
        m0 = self.get_explicit_entity_at(explicit_entity, 0)
        self.memory_content.append(content[:,0])
        self.memory_entity.append(m0)
        self.memory_location.append(tf.zeros([self.batch_size,], tf.int32))

    def feature_sentence(self, content, context_len):
        memory_text = tf.transpose(tf.nn.embedding_lookup(self.SE, self.memory_content),
                                   [1, 0, 2, 3])  # [batch * context_len * sent_len * sdim ]
        memory_text_BoW = tf.reduce_sum(memory_text, 2)  # [batch * context_len * sdim]

        query_text = tf.nn.embedding_lookup(self.SE, content[:,context_len])  # [batch * sent_len * sdim]
        query_text_BoW = tf.reduce_sum(query_text, 1)  ## [batch * sdim]
        query_text_raw_tile = tf.reshape(tf.tile(query_text_BoW, [context_len, 1]),
                                         [context_len, self.batch_size, self.sdim])
        query_text = tf.transpose(query_text_raw_tile, [1, 0, 2])
        assert (query_text.shape == (self.batch_size, context_len, self.sdim))

        get_by_cossim = False
        if get_by_cossim:
            qt_T = tf.reshape(query_text, [self.batch_size, context_len, 1, self.sdim])
            mt = tf.reshape(memory_text_BoW, [self.batch_size, context_len, self.sdim, 1])
            cossim = tf.matmul(qt_T, mt)
            assert (cossim.shape == (self.batch_size, context_len, 1, 1))
            weights_3 = tf.reshape(cossim, [self.batch_size, context_len, 1])
            return weights_3

        sent_feature = tf.concat([memory_text_BoW, query_text], 2)  # [ batch * context_len * (sdim*2)]
        QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])
        weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]
        assert_shape(weights_3, (self.batch_size, context_len, 1))
        return weights_3

    def merge_weight(self, weights_1, weights_2, weights_3, context_len):
        # Weigt Sum
        weight_concat = tf.concat([weights_1, weights_2, weights_3], 2)  # [batch * context_len * 2]
        assert_shape(weight_concat, (self.batch_size, context_len, 3))
        W_tile = tf.tile(self.W, [self.batch_size, 1])
        W_tile = tf.reshape(W_tile, [self.batch_size, 3, 1])  # [batch * 2 * 1]

        base_weights = tf.matmul(weight_concat, W_tile)  # [batch * context_len * 1]
        raw_weights = tf.scalar_mul(1. / context_len, base_weights)  # [batch * context_len * 1]
        return raw_weights

    def fetch_all_entity_at_target(self, target_idx):
        all_entity = tf.transpose(self.memory_entity, [1,0,2])  # [batch * context * edim]
        result = []
        for i in range(self.batch_size):
            idx = target_idx[i][0]
            entity_i = all_entity[i][idx]
            result.append(entity_i)
        m_result = tf.reshape(result, [self.batch_size, self.edim])
        return m_result

    def get_explicit_entity_at(self, explicit_entity, index):
        mi_ = tf.nn.embedding_lookup(self.EM, explicit_entity[:, index])  ## [batch * max_entity * edim]
        m0 = tf.reduce_sum(mi_, 1)  # [batch * dim]
        return m0

    def network(self, content, explicit_entity, target_idx):
        self.make_memory_empty()

        self.init_first_slot(content, explicit_entity)
        no_cascade = True
        print("No Cascade")
        ## TODO : too many iteration...
        for context_len in range(1, self.mem_size):
            # 1) Select relevant article
            weights_1 = tf.transpose(tf.nn.embedding_lookup(self.LE, self.memory_location), [1,0,2])  ## [batch * context_len * 1]

            dist = [context_len - x for x in self.memory_location]
            weights_2 = tf.transpose(tf.nn.embedding_lookup(self.DE, dist), [1,0,2])  ## [batch * context_len * 1]

            weights_3 = self.feature_sentence(content, context_len)

            if no_cascade:
                mi = tf.nn.embedding_lookup(self.EM, self.explicit_entity[:,:context_len])
                Cx = tf.transpose(tf.reduce_sum(mi,2), [0,2,1])  ## [batch * context_len * edim ]
            else:
                Cx = tf.transpose(tf.stack(self.memory_entity), [1, 2, 0])  ## [batch * edim * context_len]

            raw_weights = self.merge_weight(weights_1, weights_2, weights_3, context_len)
            self.last_weight = raw_weights

            m = tf.reshape(tf.matmul(Cx, raw_weights), [self.batch_size, self.edim])  ## [batch * edim]

            # 2) Updated sum of entity
            mp = m + self.get_explicit_entity_at(explicit_entity, context_len)
            mp = cap_by_one(mp, [self.batch_size, self.edim, ])

            # 3) Update
            self.memory_content.append(content[:,context_len])
            self.memory_entity.append(mp)
            self.memory_location.append(tf.constant(context_len, tf.int32, [self.batch_size,]))

        m_result = self.fetch_all_entity_at_target(target_idx)
        return m_result

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


class MemN2N_LDA(MemN2N):
    def __init__(self, config, sess):
        super(MemN2N_LDA, self).__init__(config, sess)
        self.content = tf.placeholder(tf.float32, [None, self.mem_size, self.sdim], name="content")

    def feature_sentence(self, content, context_len):
        memory_text = tf.transpose(tf.stack(self.memory_content), [1, 0, 2])  # [batch * context_len * sdim ]

        query_text = content[:, context_len]  # [batch * sdim]
        query_text_tile = tf.reshape(tf.tile(query_text, [context_len, 1]),
                                         [context_len, self.batch_size, self.sdim])
        query_text = tf.transpose(query_text_tile, [1, 0, 2])
        assert (query_text.shape == (self.batch_size, context_len, self.sdim))

        use_cossim = False
        if use_cossim:
            qt_T = tf.reshape(query_text, [self.batch_size, context_len, 1, self.sdim])
            mt = tf.reshape(memory_text, [self.batch_size, context_len, self.sdim, 1])
            cossim = tf.matmul(qt_T, mt)
            assert (cossim.shape == (self.batch_size, context_len, 1, 1))
            weights_3 = tf.reshape(cossim, [self.batch_size, context_len, 1])
        else:
            sent_feature = tf.concat([memory_text, query_text], 2)  # [ batch * context_len * (sdim*2)]
            QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])
            weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]

        assert_shape(weights_3, (self.batch_size, context_len, 1))
        return weights_3

    def init_sentence(self, test_case):
        return self.init_content([self.mem_size, self.sdim], test_case.content, 2, np.float32)


    def load_weights(self, sent_embedding=None):
        None


class MemN2N_LSTM(MemN2N):
    def __init__(self, config, sess):
        super(MemN2N_LSTM, self).__init__(config, sess)
        self.state_size = self.sdim
        with tf.variable_scope('model') as scope:
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.lstm_init = False

    def LSTM_digest(self, text):
        with tf.variable_scope('model', reuse=self.lstm_init):
            self.lstm_init = True
            outputs, states = tf.nn.dynamic_rnn(self.lstm, text, dtype=tf.float32, time_major=False)
        return outputs[:,-1]

    def feature_sentence(self, content, context_len):
        memory_text = tf.transpose(tf.nn.embedding_lookup(self.SE, self.memory_content), # [context_len * batch * sent_len * sdim]
                                   [1, 0, 2, 3])  # [batch * context_len * sent_len * sdim ]
        memory_text = tf.reshape(memory_text, [self.batch_size*context_len, self.sent_len, self.sdim])
        memory_text_LSTM = self.LSTM_digest(memory_text)
        memory_text = tf.reshape(memory_text_LSTM, [self.batch_size, context_len, self.sdim])
        query_text = tf.nn.embedding_lookup(self.SE, content[:, context_len])  # [batch * sent_len * sdim]
        query_text_LSTM = self.LSTM_digest(query_text)  ## [batch * sdim]
        query_text_raw_tile = tf.reshape(tf.tile(query_text_LSTM, [context_len, 1]),
                                         [context_len, self.batch_size, self.sdim])
        query_text = tf.transpose(query_text_raw_tile, [1, 0, 2])
        assert (query_text.shape == (self.batch_size, context_len, self.sdim))

        sent_feature = tf.concat([memory_text, query_text], 2)  # [ batch * context_len * (sdim*2)]
        QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])
        weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]
        assert_shape(weights_3, (self.batch_size, context_len, 1))
        return weights_3


class EntityInherit(MemN2N):
    def __init__(self, config, sess):
        super(EntityInherit, self).__init__(config, sess)
        self.state_size = self.sdim
        with tf.variable_scope('model') as scope:
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        self.lstm_init = False

    def init_embedding(self):
        self.SE = tf.Variable(tf.random_normal([self.nwords, self.sdim], stddev=self.init_std))
        self.QE = tf.Variable(tf.random_normal([self.sdim * 2, 1], stddev=self.init_std))
        self.W = tf.Variable(tf.constant(0.00, shape=[1, 1]), name="W")

        self.LE = self.init_LE()
        self.DE = self.init_DE()

        self.EM = self.init_EE(self.edim)
        self.a = tf.Variable(tf.zeros([1]), name="a")
        self.b = tf.Variable(tf.zeros([1]), name="b")


    def LSTM_digest(self, text):
        with tf.variable_scope('model', reuse=self.lstm_init):
            self.lstm_init = True
            outputs, states = tf.nn.dynamic_rnn(self.lstm, text, dtype=tf.float32, time_major=False)
        return outputs[:,-1]

    def feature_sentence(self, content, context_len):
        print(context_len)
        texts = tf.nn.embedding_lookup(self.SE, content)   #[memory_size * batch * sent_len * sdim ]
        prev_text = tf.transpose(texts, [1, 0, 2, 3])  # [batch * context_len * sent_len * sdim ]
        prev_text = tf.reshape(prev_text, [self.batch_size * self.mem_size, self.sent_len, self.sdim])
        prev_text = self.LSTM_digest(prev_text)  # [batch * context_len * sdim]
        prev_text = tf.reshape(prev_text, [self.batch_size, self.mem_size, self.sdim])

        cont = []
        for i in range(self.batch_size):
            idx = context_len[i][0]
            target_text = content[i,idx,:]
            cont.append(target_text)
        cont = tf.reshape(cont, [self.batch_size, self.sent_len])
        query_text = tf.nn.embedding_lookup(self.SE, cont)  # [batch * sent_len * sdim]
        query_text_LSTM = self.LSTM_digest(query_text)  ## [batch * sdim]
        query_text_raw_tile = tf.reshape(tf.tile(query_text_LSTM, [self.mem_size, 1]),
                                         [self.mem_size, self.batch_size, self.sdim])
        query_text = tf.transpose(query_text_raw_tile, [1, 0, 2])
        assert (query_text.shape == (self.batch_size, self.mem_size, self.sdim))

        sent_feature = tf.concat([prev_text, query_text], 2)  # [ batch * mem_size * (sdim*2)]
        QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])
        weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]
        assert_shape(weights_3, (self.batch_size, self.mem_size, 1))
        return weights_3
