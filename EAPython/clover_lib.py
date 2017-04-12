# -*- coding: utf-8 -*-
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

from progress.bar import *

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'


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
    with codecs.open(path, "rb", "cp949") as f:
        return [line for line in csv.reader(f)]

def load_csv_utf(path):
    with codecs.open(path, "rb", "utf-8") as f:
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

def save_list(data, path):
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')

    with codecs.open(path, "wb", 'utf-8') as f:
        for entry in data:
            f.write(entry+"\n")

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


def remove_newline_array(articles):
    result = []
    for article in articles:
        try:
            content = article[IDX_CONTENT].replace("\n"," ").replace("\r", " ")
            new_article = article[:IDX_CONTENT] + [content] + article[IDX_CONTENT+1:]
            result.append(new_article)
        except Exception as e:
            print(e)
    return result

def remove_newline(text):
    return text.replace("\n", " ").replace("\r", " ")


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


def flatten(z):
    return [y for x in z for y in x]



class ListProgress():
    def __init__(self, l):
        self.length = len(l)
        self.bar = IncrementalBar(max=50)
        self.count = 0

    def step(self):
        self.count += 1
        if self.count == self.length / 50 :
            print("-", end="")


class Batch:
    def __init__(self, batch_size):
        self.data = []
        self.bach_size = batch_size
        self.index = 0

    def enqueue(self, sample):
        self.data.append(sample)

    def has_next(self):
        return len(self.data) >= self.index + self.bach_size

    def deque(self):
        num_input = len(self.data[0])
        input = [[] for i in range(num_input)]
        for i in range(self.bach_size):
            for j in range(num_input):
                item = self.data[self.index + i][j]
                input[j].append(item)
        self.index += self.bach_size
        return input