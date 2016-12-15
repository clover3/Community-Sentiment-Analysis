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


class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

def flatten(z):
    return [y for x in z for y in x]

def load_csv(path):
    with open(path, "rb") as f:
        return [line for line in csv.reader(f)]

def save_csv(data, path):
    with open(path, "wb") as f:
        csv.writer(f).writerows(data)

def load_csv_euc_kr(path):
    with open(path, "rb") as f:
        f = UTF8Recoder(f, "cp949")
        return [line for line in csv.reader(f)]

def save_csv_euc_kr(data, path):
    with codecs.open(path, "wb", 'cp949') as f:
        csv.writer(f).writerows(data)

def load_csv_utf(path):
    with open(path, "rb") as f:
        af = UTF8Recoder(f, "utf-8")
        return [line for line in csv.reader(af)]

def save_csv_utf(data, path):
    with open(path, "rb") as f:
        af =  codecs.getwriter("utf-8")(f)
        w = csv.writer(af)
        w.writerows(data)

def save_csv_utf2(data, path):
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


## returns list[Sentence] , Sentence
def get_sentence_token(articles):
    # list[list[String]]
    sentence_list = [article[IDX_TOKENS].split('|') for article in articles]

    result = []
    for article in sentence_list:
        result.append([raw_sentence.split('/') for raw_sentence in article])

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


def output_str_list(path, str_list):
    fp= open(path, "w")
    for line in str_list:
        fp.write(line)
        fp.write("\n")
    fp.close()


def output_array(path, indexed_article):
    fp= open(path, "w")

    for article in indexed_article:
        for token in article:
            fp.write("{} ".format(token))
        fp.write("\n")
    fp.close()

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
