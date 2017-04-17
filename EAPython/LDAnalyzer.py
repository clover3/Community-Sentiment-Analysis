import threading
import gensim
from konlpy import tag
import numpy as np

num_topic = 100

LDA_PICKLE_PATH = "data\\lda"
class LDAAnalyzer:
    def __init__(self):
        self.dictionary = ""
        self.model = ""
        self.lock = threading.Lock()
        self.twitter = tag.Twitter()
        self.tokenize_cache = dict()
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
                print("Not expected")
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
        self.model = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=100, id2word=self.dictionary, passes=20)
        # lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

        self.model.save(LDA_PICKLE_PATH)

    def dist(self, text1, text2):
        t1 = self.get_topic(text1)
        t2 = self.get_topic(text2)
        return gensim.matutils.cossim(t1, t2)


if __name__ == "__main__":
    LDA = LDAAnalyzer()
    print(LDA.get_topic_vector("안녕하세요."))

