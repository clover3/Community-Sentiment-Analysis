# -*- coding: euc-kr -*-
from multiprocessing import Process, freeze_support
import threading

from clover_lib import *
from nltk.tokenize import RegexpTokenizer
import konlpy.tag
from konlpy.utils import pprint
from gensim import corpora, models
import gensim
import itertools
import random
import time
import logging


# 전체 데이터에 대해서 LDA를 돌린다.

logging.basicConfig(filename='log\\target_detection.log',level=logging.INFO)


def get_min_dist(dist_func, targets):
    dist_list = list(map(dist_func, targets))
    min_dist = min(dist_list)
    idx_min = dist_list.index(min_dist)
    return (min_dist, idx_min)


LDA_PICKLE_PATH = "model\\lda"

TFIDF_PICKLE_PATH = "model\\tfidf"
TFIDF_DICT_PICKLE_PATH = 'model\\tfidf.dict'

class TFIDFAnalyzer:
    def __init__(self):
        self.model = ""
        self.dictionary = ""

    def create_model(self, token_corpus):
        tokens = list()
        for token in token_corpus:
            tokens.append(token)

        self.dictionary = corpora.Dictionary(tokens)
        corpus = [self.dictionary.doc2bow(text) for text in tokens]
        self.model = models.TfidfModel(corpus)
        self.model.save(TFIDF_PICKLE_PATH)
        self.dictionary.save(TFIDF_DICT_PICKLE_PATH)

    def get_vector(self, article):
        tokens = article[IDX_TOKENS]
        bow = self.dictionary.doc2bow(tokens)
        return self.model[bow]

    def load_from_file(self, path):
        self.model = gensim.models.TfidfModel.load(path)
        self.dictionary = corpora.Dictionary.load(TFIDF_DICT_PICKLE_PATH)

    def dist(self, article1, article2):
        v1 = self.get_vector(article1)
        v2 = self.get_vector(article2)
        return gensim.matutils.cossim(v1,v2)


class LDAAnalyzer:
    def __init__(self):
        self.dictionary = ""
        self.model = ""
        self.lock = threading.Lock()

    def get_topic(self, article):
        self.lock.acquire()
        self.lock.release()
        tokens = []
        try:
            tokens = article[IDX_TOKENS]
        except:
            print(article)
        bow = self.dictionary.doc2bow(tokens)

        return self.model[bow]

    def load_from_file(self, path):
        self.model = gensim.models.LdaModel.load(path)
        self.dictionary = self.model.id2word

    def create_lda_model(self, text_list):
        texts = text_list

        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        # generate LDA model
        self.model = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=100, id2word = self.dictionary, passes=20)
        # lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

        self.model.save(LDA_PICKLE_PATH)

    def dist(self, text1, text2):
        t1 = self.get_topic(text1)
        t2 = self.get_topic(text2)
        return gensim.matutils.cossim(t1,t2)

class UserNotFoundError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class HeadNotFoundError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def texts_by(texts, user):
    result = []
    for text in texts:
        if text[IDX_AUTHOR_ID] == user:
            result.append(text)
    if len(result) == 0:
        raise UserNotFoundError(user)
    return result

def resolve_target(data):
    target_dict = {}

    lda_model = LDAAnalyzer()
    print("DEBUG : Loading LDA Model...")
    lda_model.load_from_file(LDA_PICKLE_PATH)

    counter = FailCounter()
    """
    Case1. Text Type = 0 => 의존 텍스트 없음
    Case2. Text Type = 1 => 해당 쓰레드의 첫글이 의존 텍스트
    Case3. Text Type = 2 첫번째 글 => 해당하는 직전 댓글이 의존 텍스트
    Case4. Text Type = 2 글 안에 아이디를 명시했을 경우 => 직전의 해당 아이디의 글이 의존 텍스트
    Case5. Text Type = 2 의존 텍스트 후보 중에 자신이 다른 사용자가 한명 뿐인 경우 => 해당 사용자의 마지막 글이 의존 텍스트
    ----
    6. Text Type = 2 그외의 경우 => 의존 텍스트 후보 중에 LDA Topic 유사도가 가장 높은 글
    """
    def find_target(list_candidate, text):
        user = text[IDX_AUTHOR_ID]
        list_filtered = list(filter(lambda x: x[IDX_AUTHOR_ID] != user, list_candidate))

        def lda_dist(text_target):
            return lda_model.dist(text_target, text)

        lda_min, lda_min_idx = get_min_dist(lda_dist, list_filtered)
        return list_candidate[lda_min_idx]

    def resolve_thread(texts):
        thread_dict = {}

        head_article_0 = -1
        head_article_1 = -1

        def get_head_article(article_id, text_type):
            if text_type == 0 :
                if head_article_0 == -1:
                    return -1
                return head_article_0
            elif text_type == 1 :
                if head_article_1 == -1:
                    return -1
                return head_article_1
            else :
                raise ValueError("invalid type : " + str(text_type))

        users = set()
        type2_count = 0
        prev = []

        for article in texts:
            target_id = -1
            article_id = article[IDX_ARTICLE_ID]
            if article[IDX_TEXT_TYPE] == '0':                ### Case 1
                target_id = 0
                type2_count = 0
                head_article_0 = article_id
            elif article[IDX_TEXT_TYPE] == '1':             ### Case 2
                target_id = get_head_article(article_id, 0)
                type2_count = 0
                head_article_1 = article_id
                prev = [article]
            elif article[IDX_TEXT_TYPE] == '2':
                def handle_case3():
                    if type2_count == 0:
                        return get_head_article(article_id, 1)
                    else:
                        return -1
                def handle_case4():
                    id = -1
                    user_ref = contain_any(article[IDX_CONTENT], users)
                    if len(user_ref) > 0:
                        try:
                            referee = texts_by(prev, user_ref)[-1]
                            id = referee[IDX_ARTICLE_ID]
                        except:
                            "None"
                    return id
                def handle_case5():
                    user = article[IDX_AUTHOR_ID]
                    list_filtered = list(filter(lambda x: x[IDX_AUTHOR_ID] != user, prev))
                    cadidate_users = set(map(lambda x:x[IDX_AUTHOR_ID], list_filtered))
                    if len(cadidate_users) == 1:
                        return list_filtered[-1][IDX_ARTICLE_ID]
                    else :
                        return -1

                target_id = handle_case3()       ### Case 3
                if target_id == -1 :
                    target_id = handle_case4()   ### Case 4
                if target_id == -1 :
                    target_id = handle_case5()

                type2_count += 1
                prev.append(article)
            else:
                raise ValueError("Unexpected Case : " + str(article[IDX_TEXT_TYPE]))

            if target_id == -1:
                try:
                    if len(prev) == 0:
                        raise ("list must not be empty")
                    user = article[IDX_AUTHOR_ID]
                    list_filtered = list(filter(lambda x: x[IDX_AUTHOR_ID] != user, prev))
                    if len(list_filtered) == 0:
                        raise ("Not found")

                    target = find_target(prev, article)
                    target_id = target[IDX_ARTICLE_ID]
                except:
                    target_id = -1

                target_dict[article_id] = target_id
                print("{} -> {}".format(article_id, target_id))
                counter.fail()
            else:
                target_dict[article_id] = target_id
                counter.suc()

            users.add(article[IDX_AUTHOR_ID])
        return thread_dict

    for key, group in itertools.groupby(data, lambda x: x[IDX_THREAD_ID]):
        texts = list(group)
        d = resolve_thread(texts)
        target_dict.update(d)

    print("Data : ", len(data))
    print("Resolved : ", len(target_dict))
    print("Precision() =", counter.precision())

    return target_dict


def test_on_guldang():
    cursor = 0000
    size = 100000
    data = load_csv_euc_kr("input\\clean_guldang_tkn.csv")[cursor:cursor+size]
    #data = load_csv_euc_kr("input\\guldang_10000_tkn.csv")
    # Format
    """
    rating
    title
    content
    authorID
    tag
    articleID
    threadID
    date
    time
    textType
    tokens
    """

    print("DEBUG : Parsing corpus...")
    data = parse_token(data)

    def print_invalid_data():
        for text in data:
            if len(text) < 9 :
                print(text)
    print_invalid_data()

    def get_text_list(csv_data):
        return list(map(lambda x: x[IDX_TOKENS], csv_data))

    corpus = get_text_list(data)

    # Init tfidf
    print("DEBUG : Creating TFIDF Model...")
    tfidf_model = TFIDFAnalyzer()
    tfidf_model.create_model(corpus)
    #tfidf_model.load_from_file("tfidf")

    # Init LDA
    lda_model = LDAAnalyzer()
    create_new = True
    if create_new:
        print("DEBUG : Creating LDA Model...")
        time1 = time.time()
        lda_model.create_lda_model(corpus)
        time2 = time.time()
        print("DEBUG : Created LDA Model (elapsed : {})".format(time2-time1))
    else:
        print("DEBUG : Loading LDA Model...")
        lda_model.load_from_file(LDA_PICKLE_PATH)

    # for each thread
    counter_lda = FailCounter()
    counter_tfidf = FailCounter()
    counter_blind = FailCounter()

    time_begin = time.time()
    print("DEBUG : Iterating threads...")
    for key, group in itertools.groupby(data, lambda x: x[IDX_THREAD_ID]):
        texts = list(group)
        analyze_thread(counter_lda, counter_tfidf, counter_blind, lda_model, tfidf_model, texts)

    print("DEBUG : Iterating Done {} elapsed".format(time.time() - time_begin))

    print("Precision(LDA) = {}".format(counter_lda.precision()))
    print("Precision(tf-idf) = {}".format(counter_tfidf.precision()))
    print("Precision(Blind) =", counter_blind.precision())
    print("Tested case : {}".format(counter_lda.total()))


def analyze_thread(counter_lda, counter_tfidf, counter_blind, lda_model, tfidf_model, texts):
    try:
        users = set(map(lambda x: x[IDX_AUTHOR_ID], texts))
        prev = []
        for text in texts:
            content = text[2]
            for user in users:
                if user in content:
                    try:
                        referrer = text
                        referee = texts_by(prev, user)[-1]

                        #  calculate dist for all texts(previous only)
                        def lda_dist(text_target):
                            return lda_model.dist(text_target, referrer)

                        def tfidf_dist(text_target):
                            return tfidf_model.dist(text_target, referrer)


                        lda_min, lda_min_idx = get_min_dist(lda_dist, prev)
                        tfidf_min, tfidf_min_idx = get_min_dist(tfidf_dist, prev)
                        counter_blind.suc()
                        counter_blind.add_fail(len(prev)-1)

                        # compare min_dist with  ref_dist
                        ref_dist = lda_model.dist(referrer, referee)

                        def fail_check(counter, min_found, idx_min):
                            if ref_dist <= min_found:
                                counter.suc()
                            else:
                                counter.fail()
                                # compare latest and text
                                logging.info("------------------")
                                logging.info("Thread {}. {} <- {}".format(referee[IDX_THREAD_ID], referee[IDX_THREAD_ID],
                                                                          referrer[IDX_ARTICLE_ID]))
                                logging.info("Real  target(Dist={}) : {}".format(ref_dist, referee[IDX_CONTENT]))
                                logging.info("Guess target(Dist={} : {}".format(min_found, prev[idx_min][IDX_CONTENT]))
                                logging.info("Referrer -> {} ".format(referrer[IDX_CONTENT]))


                        fail_check(counter_lda, lda_min, lda_min_idx)
                        fail_check(counter_tfidf, tfidf_min, tfidf_min_idx)

                    except UserNotFoundError as e:
                        abc = ""
                        logging.info("Exception : user = " + e.value)
            prev.append(text)
    except Exception as e :
        print(e)

def test_bobae():
    cursor = 0
    size = 1000
    data = load_csv_euc_kr("input\\bobae_car_tkn.csv")[cursor:cursor + size]
    data_token_parsed = parse_token(data)
    resolve_target(data_token_parsed)

if __name__ == '__main__':
    freeze_support()
    #test_on_guldang()
    test_bobae()

#create_lda_model(["썩션은 필요없구요 블로우작업은 안하시는게 낫습니다. 수분이 들어가요 문제는 수분거를만한 필터를 단 콤프레셔 사용하는 업체가 국내엔 없는걸로 알고있습니다.","golf20tdi님// 그렇죠.. 저도 에어는 절대하지말라해서 자유낙하만 했는데 충분하군요 with beta"])
