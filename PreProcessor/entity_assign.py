# -*- coding: euc-kr -*-
from types import *
from clover_lib import *
import itertools
from konlpy.tag import Kkma
from KoreanNLP import *
import pickle
import os


def remove_substring(word_list):
    word_list.sort(key=len)
    unique_words = []
    for index in range(len(word_list)):
        fAdd = True
        word = word_list[index]
        for word2 in word_list[index+1:]:
            if word in word2 :
                fAdd = False
                break
        if fAdd:
            unique_words.append(word)

    return unique_words


def match_all(content, lst):
    words = []
    for item in lst:
        l_content = content.lower()
        l_item = item.lower()
        index = l_content.find(l_item)
        if index == 0 :
            words.append(item)
        elif index > 0 :
            c = content[index-1]
            if c == '.' or c == ' ' or c == '\n' or c == '?':
                words.append(item)


    return remove_substring(words)


def contain_any(content, lst):
    for item in lst:
        if item in content:
            print(item)
            return True
    return False


def content_contain_any(text, lst):
    content = text[IDX_TITLE] + " " + text[IDX_CONTENT]
    return contain_any(content, lst)


def thread_with_keyword(data):
    s = set()
    for text in data:
        thread_id = text[6]
        if thread_id not in s and content_contain_any(text, keywords):
            s.add(thread_id)
    return s


def filter_with_threads(data, threads):
    return list(filter(lambda text: text[6] in threads, data))


def filter_article(data, given_filter):
    return list(filter(given_filter, data))

    #filtered_data = filter_with_threads(data, thread_with_keyword(data) )


def get_filtered_bobaedream():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae"]
    car_keyword = load_list("input\\car_keyword.txt")
    keywords = car_keyword

    def article_with_car_keyword(article):
        return content_contain_any(article, car_keyword)

    data = load_csv_euc_kr("input\\bobae_car.csv")

    filtered_data = filter_article(data, article_with_car_keyword)

    save_csv_euc_kr(filtered_data,"input\\bobae_articles_car.csv")
    print("Total article :" + str(len(data)))
    print("Filtered article :" + str(len(filtered_data)))


def make_corpus():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae","현기차","현대차"]
    car_keyword = load_list("input\\car_keyword.txt")
    keywords = car_keyword + keyword_hyundai
    kkma = Kkma()
    data = load_csv_euc_kr("input\\bobae_car.csv")
    corpus = []

    corpus_index = 0
    for article in data:
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        if article[IDX_TEXT_TYPE] == '0' :#and len(content) > 200 :
            continue
        #sentences = kkma.sentences(content)
        #for text in sentences:
        text = content
        words = match_all(text, keywords)
        if len(words) > 0 :
 #           found_words = ",".join(words)
 #           print("[{}] : {}".format(found_words, text))
            for word in words:
                entry = [corpus_index, article[IDX_ARTICLE_ID], article[IDX_THREAD_ID], article[IDX_CONTENT], word, 0]
                corpus.append(entry)
                corpus_index += 1

    save_csv_utf(corpus, "input\\corpus.csv")

def split_sentence_euc(str):
    if type(str)== type("str"):
        u_str = unicode(str, "euc-kr")
        u_result = split_sentence_u(u_str)
        return [s.encode("euc-kr") for s in u_result]
    else:
        return split_sentence_u(str)

def entity_assign_corpus():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae","현기차","현대차"]
    car_keyword = [str(item.encode("utf-8")) for item in load_list("..\\input\\car_keyword.txt")]

    keywords = car_keyword + keyword_hyundai
    data = load_csv("..\\input\\bobae_car_euc.csv")

    corpus = []

    def get_content(article):
        if article[IDX_TITLE]:
            content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        else:
            content = article[IDX_CONTENT]

        return content

    for article in data:
        content = get_content(article)
        for text in split_sentence_euc(content):
            words = ",".join(match_all(text, keywords))
            url = "http://www.bobaedream.co.kr/view?code=national&No="+article[IDX_THREAD_ID]
            entry = [article[IDX_ARTICLE_ID], article[IDX_THREAD_ID], text, words, article[IDX_AUTHOR_ID], url]
            corpus.append(entry)

    save_csv(corpus, "..\\input\\entity_corpus.csv")

def analyze_entity_proportion():
    car_keyword = load_list("..\\input\\car_keyword2.txt")
    keywords = car_keyword

    data = load_csv_euc_kr("..\\input\\bobae_car_euc.csv")
    corpus = []

    corpus_index = 0
    for article in data:
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        if article[IDX_TEXT_TYPE] == '0':  # and len(content) > 200 :
            continue
        # sentences = kkma.sentences(content)
        # for text in sentences:
        text = content
        words = match_all(text, keywords)
        if len(words) > 0:
            #           found_words = ",".join(words)
            #           print("[{}] : {}".format(found_words, text))
            for word in words:
                entry = [corpus_index, article[IDX_ARTICLE_ID], article[IDX_THREAD_ID], article[IDX_CONTENT], word, 0]
                corpus.append(entry)
                corpus_index += 1


    print("total article : {} / entity article : {}  : {}%".format(len(data), corpus_index, float(corpus_index)/len(data)))


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
        return load_csv(path)

    def parse_entity_label(sentences):
        for s in sentences:
            s[EL_IDX_SENTENCE_ID] = int(s[EL_IDX_SENTENCE_ID])
            assert type(s[EL_IDX_AUTHOR]) is StringType
            s[EL_IDX_ARTICLE_ID] = int(s[EL_IDX_ARTICLE_ID])
            s[EL_IDX_THREAD_ID] = int(s[EL_IDX_THREAD_ID])
            assert type(s[EL_IDX_CONTENT]) is StringType
            assert type(s[EL_IDX_AUTHOR]) is StringType


    sentences = load_entity_label(label_data_path)
    parse_entity_label(sentences)
    target_dictionary = pickle.load(open("targetDict.p","rb"))

    def get_thread_article_id(post):
        return post[EL_IDX_THREAD_ID], post[EL_IDX_ARTICLE_ID]


    ## this is inefficient
    def get_article(ta_id):
        (thread_id, article_id) = ta_id
        for s in sentences:
            tid_s = s[EL_IDX_THREAD_ID]
            aid_s = s[EL_IDX_ARTICLE_ID]
            assert type(thread_id) is IntType
            assert type(article_id) is IntType
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

        return context

    begin = sentences[0][EL_IDX_SENTENCE_ID]
    end = sentences[-1][EL_IDX_SENTENCE_ID]
    cases = range(begin, end+1)

    def generate_case(case_id):
        entity = get_sentence_by_id(case)[EL_IDX_LABEL]
        target = get_sentence_by_id(case)[EL_IDX_CONTENT]
        context = context_for(case)

        return (entity, target, context)

    return [generate_case(case) for case in cases]

def save_as_files(data):
    index = 0

    def num_lines(text):
        return text.count('\n')+1

    def save_testcase(case, index):
        f = open("entity_test\\case{}.txt".format(index), "w")
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





class solver:

    def __init__(self, keyword_path):
        keywordDict = dict()

    def solve_n_eval(self, path):
        ## for every file in dir(path) solve and summarize accuracy


    def solve(self, case):
        (entity, target, context) = case



if __name__ == '__main__':
    #entity_assign_corpus()

    save_as_files(generate_sentence_context("..\\input\\EntityLabel.csv"))