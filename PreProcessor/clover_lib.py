# -*- coding: euc-kr -*-
import csv
import codecs
import threading

IDX_RATING = 0
IDX_TITLE = 1
IDX_CONTENT = 2
IDX_AUTHOR_ID = 3
IDX_ARTICLE_ID = 5
IDX_THREAD_ID = 6
IDX_TEXT_TYPE = 9
IDX_TOKENS = 10



def flatten(z):
    return [y for x in z for y in x]

def load_csv(path):
    with open(path, "rb") as f:
        return [line for line in csv.reader(f)]

def load_csv_euc_kr(path):
    with codecs.open(path, "rb", "cp949") as f:
        return [line for line in csv.reader(f)]


def save_csv_euc_kr(data, path):
    with codecs.open(path, "wb", 'cp949') as f:
        csv.writer(f).writerows(data)

def save_csv_utf(data, path):
    with codecs.open(path, "wb", 'utf-8') as f:
        csv.writer(f).writerows(data)

def load_list(path):
    with codecs.open(path, "r", 'utf-8') as f:
        list = f.read().splitlines()
        return list


# if contain any keyword, return the keyword. else, return null string
def contain_any(text, iterable):
    for keyword in iterable:
        if keyword in text:
            return keyword
    return ""

def parse_sentence_token(articles):
    sentence_list = flatten([article[IDX_TOKENS].split('|') for article in articles])
    result = [sentence.split('/') for sentence in sentence_list]
    return result

def parse_token(articles):
    result = []
    for article in articles:
        try:
            tokens = article[IDX_TOKENS].split('/')
            n_article = article[0:IDX_TOKENS] + [tokens]
            result.append(n_article)
        except Exception as e:
            print(e)
    return result


class CaseCounter:
    def __init__(self):
        self.dic = {}

    def add_count(self, id):
        if id in self.dic:
            count = self.dic[id]
            self.dic[id] = count + 1
        else:
            self.dic[id] = 1

    def enum_count(self):
        for key in self.dic.keys():
            print("{} : {}".format(key, self.dic[key]))



class FailCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.success = 0
        self.failure = 0

    def total(self):
        return self.success + self.failure

    def suc(self):
        self.lock.acquire()
        self.success += 1
        self.lock.release()

    def fail(self):
        self.lock.acquire()
        self.failure += 1
        self.lock.release()

    def add_fail(self, count):
        self.lock.acquire()
        self.failure += count
        self.lock.release()

    def precision(self):
        total = self.success + self.failure
        return float(self.success) / total
