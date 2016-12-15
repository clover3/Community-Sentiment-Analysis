# -*- coding: euc-kr -*-
from multiprocessing import Process, freeze_support
import threading


from clover_lib import *
import gensim.models
import itertools
import random
import time
import logging


# 전체 데이터에 대해서 LDA를 돌린다.



def get_min_dist(dist_func, targets):
    dist_list = list(map(dist_func, targets))
    min_dist = min(dist_list)
    idx_min = dist_list.index(min_dist)
    return (min_dist, idx_min)


LDA_PICKLE_PATH = "model\\lda"

TFIDF_PICKLE_PATH = "model\\tfidf"
TFIDF_DICT_PICKLE_PATH = 'model\\tfidf.dict'


class Option(object):
    def __init__(self, o_input, isFail= False):
        self.fFail = isFail
        self.obj = o_input

    def isFail(self):
        return self.fFail

    def get(self):
        if self.fFail:
            raise Exception("Cannot get from failed Option")
        return self.obj

    @staticmethod
    def fail():
        return Option(None, True)



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


def texts_by(texts, user): # type : (List(text), string) -> List(text)
    result = []
    for text in texts:
        if text[IDX_AUTHOR_ID] == user:
            result.append(text)
    if len(result) == 0:
        raise UserNotFoundError(user)
    return result


def get_thread_article_id(post):
    return int(post[IDX_THREAD_ID]), int(post[IDX_ARTICLE_ID])


def resolve_target_and_eval(data):
    target_dict = {}

    lda_model = LDAAnalyzer()
    print("DEBUG : Loading LDA Model...")
    lda_model.load_from_file(LDA_PICKLE_PATH)

    case_counter = CaseCounter()
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

        head_article_0 = None  ## article stack : this is article
        head_article_1 = None  ## article stack : this is comment

        # text_type : int
        def get_current_article():
            return head_article_0

        def get_current_comment():
            return head_article_1

        users = set()
        prev_coc = []

        for post in texts:
            if post[IDX_TEXT_TYPE] not in ['0','1','2']:
                raise ValueError("Unexpected Case : " + str(post[IDX_TEXT_TYPE]))

            # update stack info
            if post[IDX_TEXT_TYPE] == '0':
                head_article_0 = post
                prev_coc = []
                users = set()
            elif post[IDX_TEXT_TYPE] == '1':
                head_article_1 = post
                prev_coc = []

            def handle_case1():
                if post[IDX_TEXT_TYPE] == '0':  ### Case 1
                    case_counter.add_count(1)
                    return Option((0,0))
                else:
                    return Option.fail()

            def handle_case2():
                if post[IDX_TEXT_TYPE] == '1':  ### Case 2
                    case_counter.add_count(2)
                    return Option(get_thread_article_id(get_current_article()))
                else:
                    return Option.fail()

            def handle_case3():  # () -> Option((int,int))
                if post[IDX_TEXT_TYPE] != '2':
                    return Option.fail()

                if len(prev_coc) == 0 :
                    case_counter.add_count(3)
                    if get_current_comment() is None:
                        return Option((0,0))
                    return Option(get_thread_article_id(get_current_comment()))
                else:
                    return Option.fail()

            def handle_case4():
                if post[IDX_TEXT_TYPE] != '2':
                    return Option.fail()
                id = Option.fail()
                user_ref = contain_any(post[IDX_CONTENT], users)
                if len(user_ref) > 0:
                    try:
                        case_counter.add_count(4)
                        candidates = list(prev_coc)
                        if get_current_comment() is not None:
                            candidates.insert(0, get_current_comment())
                        referee = texts_by(candidates, user_ref)[-1]  # type : text
                        id = Option(get_thread_article_id(referee))
                    except:
                        "None"
                return id

            def handle_case5():
                if post[IDX_TEXT_TYPE] != '2':
                    return Option.fail()

                user = post[IDX_AUTHOR_ID]
                candidates = list(prev_coc)

                if get_current_comment() is not None:
                    candidates.insert(0, get_current_comment())

                list_filtered = list(filter(lambda x: x[IDX_AUTHOR_ID] != user, candidates))
                candidate_users = set(map(lambda x: x[IDX_AUTHOR_ID], list_filtered))
                if len(candidate_users) == 1:
                    text = list_filtered[-1]
                    case_counter.add_count(5)
                    return Option(get_thread_article_id(text))
                else:
                    return Option.fail()

            def handle_case6():
                counter.fail()
                try:
                    if len(prev_coc) == 0:
                        raise Exception("list must not be empty")
                    user = post[IDX_AUTHOR_ID]
                    candidates = list(prev_coc)
                    if get_current_comment():
                        candidates.insert(0, get_current_comment())

                    list_filtered = list(filter(lambda x: x[IDX_AUTHOR_ID] != user, candidates))
                    if len(list_filtered) == 0:
                        raise Exception("Not found")

                    case_counter.add_count(6)
                    target = find_target(candidates, post)
                    return Option(get_thread_article_id(target))
                except Exception as e:
                    print(e)
                    return Option((0,0))

            ret = handle_case1()       ### Case 3
            if ret.isFail():
                ret = handle_case2()
            if ret.isFail():
                ret = handle_case3()
            if ret.isFail():
                ret = handle_case4()
            if ret.isFail():
                ret = handle_case5()
            if ret.isFail():
                ret = handle_case6()
            else:
                counter.fail()

            target_dict[get_thread_article_id(post)] = ret.get()

            if post[IDX_TEXT_TYPE] == '2':
                prev_coc.append(post)

            users.add(post[IDX_AUTHOR_ID])

        return thread_dict

    for key, group in itertools.groupby(data, lambda x: x[IDX_THREAD_ID]):
        texts = list(group)
        d = resolve_thread(texts)
        target_dict.update(d)

    print("Data : ", len(data))
    print("Resolved : ", len(target_dict))
    print("Precision() =", counter.precision())
    case_counter.enum_count()

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

def validate(data): # type : List() -> ()
    for entry in data:
        assert(int(entry[IDX_ARTICLE_ID]) > 0)
        assert(int(entry[IDX_THREAD_ID]) > 0 )
        ttype = entry[IDX_TEXT_TYPE]
        assert (ttype =='0' or ttype == '1' or ttype == '2')
        assert(type(entry[IDX_TOKENS])==type([1,2,3]))

def test_bobae():
    cursor = 0
    size = 1000
    data = load_csv_utf("..\\input\\bobae_car_tkn_twitter.csv")[cursor:cursor + size]

    data_token_parsed = parse_token(data)
    validate(data_token_parsed)
    resolve_target_and_eval(data_token_parsed)


IDX_R_THREAD = 1
IDX_R_ARTICLE = 2
IDX_R_KEYWORD = 3
IDX_R_LABEL = 4

def validate_a_contain_b(a, b):
    a_set = set([get_thread_article_id(line) for line in a ])

    def get_thread_article_id_r(post):
        return int(post[IDX_R_THREAD]), int(post[IDX_R_ARTICLE])

    allTrue = all(get_thread_article_id_r(elem) in a_set for elem in b)

    assert allTrue


def load_idx2word(path):
    data = codecs.open(path, "r", encoding="cp949").readlines()
    idx2word = dict()
    word2idx = dict()
    for line in data:
        idx = int(line.split()[0])
        raw_word =(line.split())[1].strip()
        assert type(raw_word)==type(u"hello")

        word = raw_word.encode("utf-8")

        assert type(word)==type("word")
        idx2word[idx] = word
        word2idx[word] = idx
    return idx2word, word2idx


def load_rules(path, word2idx):
    data = codecs.open(path, "r", encoding="cp949").readlines()

    def transform(line):
        item = int(line.split()[0])
        condition = int(line.split())[1]
        return item,condition

    return [transform(line) for line in data]


def to_indexed_string(tokens, word2idx):
    real_tokens = filter(lambda x: len(x) > 0, tokens)
    try:
        index_tokens = [str(word2idx[word]) for word in real_tokens]
    except KeyError as e:
        print e.args[0]
        raise e
    if len(index_tokens) == 0:
        index_tokens = ['5']
    return " ".join(index_tokens)


def extract_pair(data, related_dic, word2idx): # List(text), Dic((int,int) -> (int,int)) -> List(String)
    data_dic = dict() # Dict( (int,int) -> text )

    for post in data:
        index = get_thread_article_id(post)
        data_dic[index] = post

    def post2pair_str(post):
        index = get_thread_article_id(post)
        related_index = related_dic[index]
        out_str = to_indexed_string(post[IDX_TOKENS], word2idx)
        if related_index == (0,0):
            None
        else :
            related_post = data_dic[related_index] # text
            post_fix = to_indexed_string(related_post[IDX_TOKENS], word2idx)
            out_str += (" - " + post_fix)
        return out_str

    return [post2pair_str(post) for post in data]



def process_bobae():

    idx2word, word2idx = load_idx2word("..\\input\\idx2word")

    cursor = 0
    size = 100000
    data = load_csv_utf("..\\input\\bobae_car_tkn_twitter.csv")[cursor:cursor + size]

    data_token_parsed = parse_token(data)
    validate(data_token_parsed)
    related_dic = resolve_target_and_eval(data_token_parsed)

    label_path = "..\\input\\corpus_samba.csv"
    labels = load_csv(label_path)

    validate_a_contain_b(data_token_parsed, labels)

    ###  Sentence 1 - Related Sentence N
    str_list = extract_pair(data_token_parsed, related_dic, word2idx)
    output_str_list("..\\input\\related.index", str_list)


def load_recovered(path, idx2word):
    raw_list = load_list(path)

    def line2int_list(line):
        return [idx2word[int(token)] for token in line.split()]

    return [line2int_list(line) for line in raw_list]



def replace_token(articles, replace):
    result = []
    for i,article in enumerate(articles):
        try:
            tokens = article[IDX_TOKENS].split('/')
            n_article = article[0:IDX_TOKENS] + [replace[i]]
            result.append(n_article)
        except Exception as e:
            print(e)
    return result

def apply_recovered():
    idx2word, word2idx = load_idx2word("..\\input\\idx2word")
    cursor = 0
    size = 100000
    data = load_csv_utf("..\\input\\bobae_car_tkn_twitter.csv")[cursor:cursor + size]
    recovered = load_recovered("..\\input\\recovered.index", idx2word)
    lined = ["/".join(tokens) for tokens in recovered]

    print type(lined[0])
    print lined[0]

    data_token_parsed = replace_token(data, lined)
    save_csv(data_token_parsed, "babae_car_recovered.csv")



if __name__ == '__main__':
    freeze_support()
    #test_on_guldang()

    #process_bobae()
    apply_recovered()
#create_lda_model(["썩션은 필요없구요 블로우작업은 안하시는게 낫습니다. 수분이 들어가요 문제는 수분거를만한 필터를 단 콤프레셔 사용하는 업체가 국내엔 없는걸로 알고있습니다.","golf20tdi님// 그렇죠.. 저도 에어는 절대하지말라해서 자유낙하만 했는데 충분하군요 with beta"])
