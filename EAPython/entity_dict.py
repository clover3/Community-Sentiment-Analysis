from clover_lib import *

Nothing = ""


class EntityDict:
    def __init__(self, dict_path):
        lines = open(dict_path, encoding='UTF8').readlines()

        def parse_line(line):
            token = line.strip().split("\t")
            group_num = int(token[0].strip())
            raw_entitys = token[1: ]
            non_empty_entity = filter(lambda x: len(x) > 0, raw_entitys)
            pairs = [(entity.lower(), group_num) for entity in non_empty_entity]
            return pairs

        def gen_group_name(line):
            token = line.strip().split("\t")
            group_num = int(token[0].strip())
            entity= token[1].strip()
            return group_num, entity

        list_group_id = [parse_line(line) for line in lines]
        keyword_pair = flatten(list_group_id)

        self.entity_list = [x[0] for x in keyword_pair]
        self.cache = dict()
        self.entity2group = dict(keyword_pair)
        self.id2names = dict([gen_group_name(line) for line in lines])

    def is_start_of_token(self, c):
        return c in ['.', ' ', '\n', '?']

    def langInversion(self, c, c2):
        return not c.isalpha() and c2.isalpha()

    ## return : list of entity
    def extract_from(self, sentence):
        if sentence in self.cache:
            return self.cache[sentence]

        def get_if_exists(string, pattern):
            idx = string.find(pattern)
            if idx < 0 :
                return Nothing
            elif idx == 0:
                return pattern
            elif self.is_start_of_token(string[idx-1]) :
                return pattern
            else:
                return get_if_exists(string[idx+1:], pattern)

        list = []
        for entity in self.entity_list:
            res = get_if_exists(sentence.lower(), entity)
            if res :
                list.append(res)

        self.cache[sentence] = list
        return list

    def get_group(self, entity):
        return self.entity2group[entity.lower()]

    def get_name(self, entity_id):
        return self.id2names[entity_id]
