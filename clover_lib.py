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


def load_csv_euc_kr(path):
    with codecs.open(path, "rb", "cp949") as f:
        return [line for line in csv.reader(f)]


def save_csv_euc_kr(data, path):
    with codecs.open(path, "wb", 'cp949') as f:
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
