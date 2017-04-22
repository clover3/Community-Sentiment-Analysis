import threading
import time
import gensim
from gensim import corpora
from konlpy import tag
import numpy as np
from clover_lib import *

num_topic = 100

LDA_PICKLE_PATH = "data\\lda"
class LDAAnalyzer:
    def __init__(self, new = False):
        self.dictionary = ""
        self.model = ""
        self.lock = threading.Lock()
        self.twitter = tag.Twitter()
        self.tokenize_cache = dict()
        if not new:
            self.load_from_file(LDA_PICKLE_PATH)
        self.topic_cache = dict()

    def tokenize(self, str):
        if str in self.tokenize_cache:
            return self.tokenize_cache[str]
        else:
            poses = self.twitter.pos(str, True, True)
            res = [pos[0] for pos in poses]
            self.tokenize_cache[str] = res
            return res

    def get_topic(self, text):
        hash = "".join(text)
        if hash in self.topic_cache:
            return self.topic_cache[hash]
        self.lock.acquire()
        self.lock.release()
        tokens = []
        try:
            if type(text) == type("str"):
                #print("Not expected")
                tokens = self.tokenize(text)
            elif type(text) == type(["List","Of","Token"]):
                tokens = text
        except:
            print(text)
        bow = self.dictionary.doc2bow(tokens)
        topic = self.model[bow]
        self.topic_cache[hash] = topic
        return topic

    def get_topic_vector(self, text):
        topics = self.get_topic(text)
        arr = [0] * num_topic
        for topic in topics:
            topic_num, prob = topic
            arr[topic_num] = prob
        return arr


    def load_from_file(self, path):
        self.model = gensim.models.LdaModel.load(path)
        self.dictionary = self.model.id2word

    def create_lda_model(self, text_list):
        texts = text_list

        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        # generate LDA model
        self.model = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=20, id2word=self.dictionary, passes=20)
        # lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

        self.model.save(LDA_PICKLE_PATH)

    def dist(self, text1, text2):
        t1 = self.get_topic(text1)
        t2 = self.get_topic(text2)
        return gensim.matutils.cossim(t1, t2)

def cossim(v1, v2):
    sum = 0
    for (a,b) in zip(v1,v2):
        sum += a*b
    return sum


def build_LDA():
    cursor = 0000
    size = 1000000
    all_data = load_csv_euc_kr("..\\input\\clean_guldang_tkn.csv")
    print("data size : {}".format(len(all_data)))
    data = all_data[cursor:cursor + size]
    print("DEBUG : Parsing corpus...")
    data = parse_token(data)

    def print_invalid_data():
        for text in data:
            if len(text) < 9:
                print(text)

    print_invalid_data()

    def get_text_list(csv_data):
        return list(map(lambda x: x[IDX_TOKENS], csv_data))

    corpus = get_text_list(data)
    lda = LDAAnalyzer(True)
    print("DEBUG : creating model...")
    start = time.time()
    lda.create_lda_model(corpus)
    elapsed = time.time() - start
    print("Done. {} elapsed ".format(elapsed))


if __name__ == "__main__":
    build_LDA()

    LDA = LDAAnalyzer()
    sentences = ["지난 제네바 모터쇼에서 처음 니로를 접하고 2열 시트에 앉았을 때의 경험이 다시금 떠올랐다.",
        "국내 뿐만 아니라 다른 어떤 메이커들의 동급 소형 SUV도 이정도의 공간을 보여주진 못했다.",
        "머리위로는 주먹 하나가 충분히 들어갈 만큼의 공간이 확보되어 있고 무릎 공간 또한 넉넉하다.",
        "2열 에어밴트 하단에는 220V 콘센트가 눈에 띈다.",
        "IT 컨비니언스 옵션을 선택한 경우 220V 콘센트와 스마트폰 무선충전, 센터콘솔의 추가 USB포트가 더해진다."]
    v = [LDA.get_topic_vector(s) for s in sentences]

    for i in range(len(v)-1):
        print(cossim(v[i], v[i+1]))
