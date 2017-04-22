# -*- coding: utf-8 -*-
import os
from clover_lib import *
from idx2word import Idx2Word
from entity_dict import EntityDict

import pickle
from LDAnalyzer import LDAAnalyzer

class TestCaseStr:
    def __init__(self, lines, entity_dict):
        line_thread_id = lines[0]
        thread_id = int(line_thread_id)

        line_raw_entity = lines[1]

        def parse_entity(raw_entity):
            if raw_entity.strip() == "-":
                return []
            else:
                return [item.strip() for item in raw_entity.split(",")]

        target_len = int(lines[2])
        target = "".join(lines[3: 3+target_len])

        def parse_context(lines):
            if len(lines) == 0:
                return []
            else:
                context_len = int(lines[0])
                context = "".join(lines[1:1+context_len])
                return [context] + parse_context(lines[1+context_len:])

        context = parse_context(lines[3+target_len:])

        self.thread_id = thread_id
        self.content = context + [target]
        self.explicit_entity = [entity_dict.extract_from(sent) for sent in self.content]

        # golden entity of last text
        self.real_entity = parse_entity(line_raw_entity)

        # both content and explicit_entity should have save length
        assert (len(self.content) == len(self.explicit_entity))


class TestCase:
    def __init__(self, thread_id, content, explicit_entity, entity):
        assert(len(content) == len(explicit_entity))
        self.thread_id = thread_id
        self.content = content
        self.explicit_entity = explicit_entity
        self.real_entity = entity

        for entity in self.explicit_entity[-1]:
            if entity not in self.real_entity:
                raise "Damn"

from konlpy.tag import Twitter


class DataSet:
    def __init__(self, data_path, entity_dict, word2idx, token_vectorizer = None):
        print("Lodaing DataSet from {} ...".format(data_path), end='', flush=True)
        ## load data path folder
        twitter = Twitter()
        tokenize_cache = dict()

        def tokenize(str):
            if str in tokenize_cache:
                return tokenize_cache[str]
            else:
                poses = twitter.pos(str, True, True)
                res = [pos[0] for pos in poses]
                tokenize_cache[str] = res
                return res

        self.test_cases_str = []

        for dirname, dirnames, filenames in os.walk(data_path):
            for filename in filenames:
                test_case = TestCaseStr(open(dirname + "\\" + filename).readlines(), entity_dict)
                self.test_cases_str.append(test_case)

        max_depth = 0
        self.test_cases = []
        for test_case_str in self.test_cases_str:
            sentences = test_case_str.content
            if len(sentences) > max_depth:
                max_depth = len(sentences)
            tokens_list = [tokenize(str) for str in sentences]

            def make_index(tokens):
                return [word2idx.word2idx(token) for token in tokens]
            if token_vectorizer:
                content = [token_vectorizer(tokens) for tokens in tokens_list]
            else:
                content = [make_index(tokens) for tokens in tokens_list]

            def entitys2index(entitys):
                es = set([entity_dict.get_group(e) for e in entitys])
                return list(es)

            explicit_entity = [entitys2index(e) for e in test_case_str.explicit_entity]

            entity_set = set(entitys2index(test_case_str.real_entity))
            entity_set.update(explicit_entity[-1])
            entity = list(entity_set)

            try:
                self.test_cases.append(TestCase(test_case_str.thread_id, content, explicit_entity, entity))
            except:
                print("---------------")
                print(sentences[-1])

                real = ",".join([entity_dict.get_name(e) for e in entity])
                explicit = ",".join([entity_dict.get_name(e) for e in explicit_entity[-1]])
                print("{} should have {}".format(real, explicit))
                raise



        print("{} data loaded".format(len(self.test_cases)))
        print("Max depth : " + str(max_depth))

def data_loader_tester():
    path = "..\\input\\entity_test"
    dict_path = "..\\input\\EntityDict.txt"
    idx2word = Idx2Word("data\\idx2word")
    entity_dict = EntityDict(dict_path)

    loader = DataSet(path, entity_dict, idx2word)

    for test_case in loader.test_cases_str:
        print("<< Test Case >>")
        n = len(test_case.content)
        for idx in range(n):
            print("{} : {}".format(test_case.content[idx], test_case.explicit_entity[idx]))

        print("Target Entity : {}".format(test_case.real_entity))


def convert_data2pickle(data_path, dict_path, pickle_path):
    print("convert_data2pickle({},{})".format(data_path, pickle_path))

    entity_dict = EntityDict(dict_path)
    idx2word = Idx2Word("data\\idx2word")
    data_set = DataSet(data_path, entity_dict, idx2word)
    pickle.dump(data_set.test_cases, open(pickle_path, "wb"))
    print("{}% words not found ".format(float(idx2word.not_found) / idx2word.trial))


def convert_data_list_2pickle(data_path_list, dict_path, pickle_path, lda_vectorizer = None):

    entity_dict = EntityDict(dict_path)
    idx2word = Idx2Word("data\\idx2word")

    if lda_vectorizer:
        sets = [DataSet(path, entity_dict, idx2word, lda_vectorizer) for path in data_path_list]
    else:
        sets = [DataSet(path, entity_dict, idx2word) for path in data_path_list]

    cases = []
    for data_set in sets:
        cases += data_set.test_cases
    all_thread = set([test_case.thread_id for test_case in cases])
    pickle.dump(cases, open(pickle_path, "wb"))

    total_case = len(cases)
    num_thread = len(all_thread)
    print("total case : {} , total thread : {}".format(total_case, num_thread))
 #   print("{}% words not found ".format(float(idx2word.not_found) / idx2word.trial))



def gen_minimal():
    dict_path = "..\\input\\EntityDict.txt"
    entity_dict = EntityDict(dict_path)


def get_lda_vectorizer():
    lda = LDAAnalyzer()
    return lda.get_topic_vector


if __name__ == "__main__":
    #data_loader_tester()

    #convert_data2pickle("data\\entity_test1", "data\\minEntityDict.txt", "data\\dataSet1_s.p")
    #convert_data2pickle("data\\entity_test3", "data\\minEntityDict.txt", "data\\dataSet3_s.p")
    convert_data_list_2pickle(["data\\entity_test1", "data\\entity_test3"], "data\\minEntityDict.txt", "data\\dataSet_lda_2.p", get_lda_vectorizer())