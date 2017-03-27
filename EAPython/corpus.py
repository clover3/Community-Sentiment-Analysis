# -*- coding: euc-kr -*-
from types import *
from clover_lib import *
import pickle
import os


def generate_sentence_context(label_data_path):
    # list of [sentenceID, articleID, threadID, content, author ]
    EL_IDX_SENTENCE_ID = 0
    EL_IDX_ARTICLE_ID  = 1
    EL_IDX_THREAD_ID   = 2
    EL_IDX_CONTENT     = 3
    EL_IDX_LABEL       = 4
    EL_IDX_AUTHOR      = 5
    NULL_IDX = (0,0)

    def load_entity_label(path):
        return load_csv_euc_kr(path)

    def parse_entity_label(sentences):
        for s in sentences:
            s[EL_IDX_SENTENCE_ID] = int(s[EL_IDX_SENTENCE_ID])
            s[EL_IDX_ARTICLE_ID] = int(s[EL_IDX_ARTICLE_ID])
            s[EL_IDX_THREAD_ID] = int(s[EL_IDX_THREAD_ID])

    sentences = load_entity_label(label_data_path)
    parse_entity_label(sentences)
    target_dictionary = pickle.load(open("data\\targetDict.p","rb"))

    def get_thread_article_id(post):
        return post[EL_IDX_THREAD_ID], post[EL_IDX_ARTICLE_ID]


    ## this is inefficient
    def get_article(ta_id):
        (thread_id, article_id) = ta_id
        for s in sentences:
            tid_s = s[EL_IDX_THREAD_ID]
            aid_s = s[EL_IDX_ARTICLE_ID]
            if tid_s == thread_id and aid_s == article_id:
                return s
        raise KeyError

    def get_author(ta_id):
        return get_article(ta_id)[EL_IDX_AUTHOR]

    def get_article_sentence(ta_id):
        result = []
        (thread_id, article_id) = ta_id

        for s in sentences:
            tid_s = s[EL_IDX_THREAD_ID]
            aid_s = s[EL_IDX_ARTICLE_ID]
            if tid_s == thread_id and aid_s == article_id:
                result.append(s[EL_IDX_CONTENT])
        return result

    def get_article_sentence_before(ta_id, max_id):
        result = []
        (thread_id, article_id) = ta_id

        for s in sentences:
            tid_s = s[EL_IDX_THREAD_ID]
            aid_s = s[EL_IDX_ARTICLE_ID]
            if tid_s == thread_id and aid_s == article_id and s[EL_IDX_SENTENCE_ID] < max_id:
                result.append(s[EL_IDX_CONTENT])
        return result

    def get_sentence_by_id(sid):
        for s in sentences:
            if int(s[EL_IDX_SENTENCE_ID]) == sid:
                return s
        raise KeyError

    def context_for(sentenceID):
        # return : List[Sentence,Author]
        sentence_structure = get_sentence_by_id(sentenceID)
        ta_id = get_thread_article_id(sentence_structure)
        context_articles = []
        while ta_id != NULL_IDX and ta_id in target_dictionary:
            target = target_dictionary[ta_id]
            if target == NULL_IDX:
                break
            context_articles.append(target)
            ta_id = target

        # Now context_articles = [prev_comment2, prev_comment1,  ... , article]
        context_articles.reverse()
        # Now context_articles = [article, prev_comment, prev_comment2,  ... , ]

        context = []
        for ta_id in context_articles:
            author = get_author(ta_id)
            for sentence in get_article_sentence(ta_id):
                context.append((sentence, author))

        # Now add sentence of current post
        ta_id = get_thread_article_id(sentence_structure)
        author = get_author(ta_id)
        for sentence in get_article_sentence_before(ta_id, sentenceID):
            context.append((sentence, author))

        return context

    begin = sentences[0][EL_IDX_SENTENCE_ID]
    end = sentences[-1][EL_IDX_SENTENCE_ID]
    cases = range(begin, end+1)

    def generate_case(case_id):
        entity = get_sentence_by_id(case_id)[EL_IDX_LABEL]
        target = get_sentence_by_id(case_id)[EL_IDX_CONTENT]
        context = context_for(case_id)

        return (entity, target, context)

    return [generate_case(case) for case in cases]


def get_all_entity(label_data_path):
    def load_entity_label(path):
        return load_csv_euc_kr(path)

    sentences = load_entity_label(label_data_path)
    EL_IDX_LABEL = 4
    labels = [s[EL_IDX_LABEL] for s in sentences]

    labels = flatten([line.split(",") for line in labels])
    labels = [l.strip() for l in labels]
    return labels



def save_as_files(data, dirPath):
    index = 0

    def num_lines(text):
        return text.count('\n')+1

    def save_testcase(case, index):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        f = open(dirPath +"\\case{}.txt".format(index), "w")
        (entity, target, context) = case
        f.write(entity+"\n")
        f.write(str(num_lines(target))+"\n")
        f.write(target+"\n")

        for token in context:
            sentence = token[0]
            f.write(str(num_lines(sentence))+ "\n")
            f.write(sentence + "\n")
            author = token[1]
            ## Note that author is not printed

    ## Entity
    ## Target Sentence Length
    ## Target Sentence
    ## Number of Context Sentence
    ## Legnth of each sentence

    for case in data:
        save_testcase(case, index)
        index += 1


def gen_simple_entity():
    s1 = get_all_entity("data\\EntityLabel.csv")
    s2 = get_all_entity("data\\EntityLabelSet2.csv")
    appeared_keyword = set(s1+s2)

    lines = open("..\\input\\EntityDict.txt", encoding='UTF8').readlines()

    def parse(line):
        token = line.strip().split("\t")
        group_num = int(token[0].strip())
        entity = [e.strip() for e in token[1:]]
        return group_num, entity

    all_entity = [parse(line) for line in lines]
    def appear(pair):
        entity = pair[1]
        for e in entity:
            if e in appeared_keyword:
                return True
            else:
                print(e)


        return False


    important = filter(appear, all_entity)

    f = open("data\\minEntityDict.txt", "w", encoding="UTF8")
    for (group_num, entity) in important:
        f.write("\t".join([str(group_num)] + entity) + "\n")



if __name__ == '__main__':
    #r = generate_sentence_context("data\\EntityLabel.csv")
    #save_as_files(r, "data\\entity_test1")

    #r = generate_sentence_context("data\\EntityLabelSet2.csv")
    #save_as_files(r, "data\\entity_test2")

    gen_simple_entity()