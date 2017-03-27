# -*- coding: utf-8 -*-
import os
from clover_lib import *
from idx2word import Idx2Word
from entity_dict import EntityDict


class TestCaseStr:
    def __init__(self, lines, entity_dict):
        line_raw_entity = lines[0]

        def parse_entity(raw_entity):
            if raw_entity.strip() == "-":
                return []
            else:
                return [item.strip() for item in raw_entity.split(",")]

        target_len = int(lines[1])
        target = "".join(lines[2: 2+target_len])

        def parse_context(lines):
            if len(lines) == 0:
                return []
            else:
                context_len = int(lines[0])
                context = "".join(lines[1:1+context_len])
                return [context] + parse_context(lines[1+context_len:])

        context = parse_context(lines[2+target_len:])

        self.content = context + [target]
        self.explicit_entity = [entity_dict.extract_from(sent) for sent in self.content]

        # golden entity of last text
        self.real_entity = parse_entity(line_raw_entity)


        # both content and explicit_entity should have save length
        assert (len(self.content) == len(self.explicit_entity))


class TestCase:
    def __init__(self, content, explicit_entity, entity):
        assert(len(content) == len(explicit_entity))
        self.content = content
        self.explicit_entity = explicit_entity
        self.real_entity = entity

        for entity in self.explicit_entity[-1]:
            if entity not in self.real_entity:
                raise "Damn"

from konlpy.tag import Twitter

class DataSet:
    def __init__(self, data_path, entity_dict, word2idx):
        print("Lodaing DataSet from {} ...".format(data_path), end='', flush=True)
        ## load data path folder
        twitter = Twitter()
        tokenize_cache = dict()

        def tokenize(str):
            if str in tokenize_cache:
                return tokenize_cache[str]
            else:
                poses = twitter.pos(str, True, True)
                res = [pos[0]+pos[1] for pos in poses]
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
            content = [make_index(tokens) for tokens in tokens_list]

            def entitys2index(entitys):
                es = set([entity_dict.get_group(e) for e in entitys])
                return list(es)

            explicit_entity = [entitys2index(e) for e in test_case_str.explicit_entity]


            entity_set = set(entitys2index(test_case_str.real_entity))
            entity_set.update(explicit_entity[-1])
            entity = list(entity_set)

            try:
                self.test_cases.append(TestCase(content, explicit_entity, entity))
            except:
                print("---------------")
                print(sentences[-1])

                real = ",".join([entity_dict.get_name(e) for e in entity])
                explicit = ",".join([entity_dict.get_name(e) for e in explicit_entity[-1]])
                print("{} should have {}".format(real, explicit))



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


def convert_data2pickle(data_path, pickle_path):
    print("convert_data2pickle({},{})".format(data_path, pickle_path))
    dict_path = "data\\minEntityDict.txt"
    entity_dict = EntityDict(dict_path)
    idx2word = Idx2Word("data\\idx2word")

    data_set = DataSet(data_path, entity_dict, idx2word)
    import pickle
    pickle.dump(data_set.test_cases, open(pickle_path, "wb"))


def gen_minimal():
    dict_path = "..\\input\\EntityDict.txt"
    entity_dict = EntityDict(dict_path)


if __name__ == "__main__":
    #data_loader_tester()
    convert_data2pickle("data\\entity_test1", "data\\dataSet1.p")
    convert_data2pickle("data\\entity_test2", "data\\dataSet2.p")