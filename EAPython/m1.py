import numpy as np
import tensorflow as tf

from model_common import *
from easolver import EASolver


class M1(EASolver):
    def __init__(self, config, sess):
        super(M1, self).__init__(config, sess)

    def init_embedding(self):
        super(M1, self).init_embedding()
        self.a = tf.Variable(tf.zeros([1]), name="a")
        self.b = tf.Variable(tf.zeros([1]), name="b")

    def feature_sentence(self, content, target_idx):
        prev_text = tf.nn.embedding_lookup(self.SE, content)   # [batch * max_sent * sent_len * sdim ]
        prev_text = tf.reduce_sum(prev_text, 2) # [batch * max_sent * sdim]

        target_texts = []
        for i in range(self.batch_size):
            idx = target_idx[i][0]
            target_text = content[i, idx, :]
            target_texts.append(target_text)

        target_texts = tf.reshape(target_texts, [self.batch_size, self.sent_len])
        query_text = tf.nn.embedding_lookup(self.SE, target_texts)  # [batch * sent_len * sdim]
        query_text_BoW = tf.reduce_sum(query_text, 1)  # [batch * sdim]
        query_text_raw_tile = tf.reshape(tf.tile(query_text_BoW, [self.max_sent, 1]),
                                         [self.max_sent, self.batch_size, self.sdim])
        query_text = tf.transpose(query_text_raw_tile, [1, 0, 2]) # [batch * max_sent * sdim]

        print_shape("prev_text:", prev_text)
        assert (query_text.shape == (self.batch_size, self.max_sent, self.sdim))
        sent_feature = tf.concat([prev_text, query_text], 2)  # [ batch * mem_size * (sdim*2)]

        QE_tile = tf.reshape(tf.tile(self.QE, [self.batch_size, 1]), [self.batch_size, self.sdim * 2, 1])
        weights_3 = tf.matmul(sent_feature, QE_tile)  # [ batch * context_len ]
        assert_shape(weights_3, (self.batch_size, self.max_sent, 1))
        return weights_3

    def network(self, content, explicit_entity, target_idx):
        ## TODO : too many iteration...
        context_len = target_idx
        weights_3 = self.feature_sentence(content, context_len)
        raw_weights = tf.sigmoid(self.a * weights_3 + self.b)
        mask = tf.transpose(raw_weights, [0,2,1]) # [ batch * 1 * mem_size ]

        mi_ = tf.nn.embedding_lookup(self.EM, explicit_entity)  ## [batch * mem_size * max_entity * edim]
        # em = entity_mask
        em_all = tf.reduce_sum(mi_, 2)  # [batch * mem_size * edim]
        em_active = tf.reshape(tf.matmul(mask, em_all), [self.batch_size, self.edim])  ## [batch * edim]

        em_explicit = []
        for i in range(self.batch_size):
            idx = context_len[i][0]
            ee = em_all[i, idx]
            em_explicit.append(ee)
        em_explicit = tf.stack(em_explicit)
        # 2) Updated sum of entity
        mp = em_active + em_explicit
        m_result = cap_by_one(mp, [self.batch_size, self.edim, ])

        return m_result
