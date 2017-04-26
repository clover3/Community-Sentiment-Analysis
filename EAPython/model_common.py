from clover_lib import *
import math
import random
import numpy as np
import tensorflow as tf
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


def cap_by_one(tensor, shape):
    ones = tf.ones(shape)
    tensor = tf.minimum(tensor, ones)
    return tensor


def probability_and(tensor, axis):
    return tf.reduce_sum(tensor, axis)
    one = tf.ones_like(tensor)
    cap = tf.minimum(tensor, one)
    nots = one - cap

    not_prob = tf.reduce_prod(nots, axis)
    ones = tf.ones_like(not_prob)
    return ones - not_prob


def activate_label(prob_tensor):
    bias_value = -0.5
    bias = tf.constant(bias_value, dtype=tf.float32, shape=prob_tensor.shape)
    return tf.round(tf.sigmoid(tf.add(prob_tensor, bias)))

