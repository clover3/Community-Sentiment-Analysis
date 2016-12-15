# -*- coding: euc-kr -*-
import codecs
from clover_lib import *
from konlpy.tag import Kkma
kkma = Kkma()

def contain_tag(line):
    return line[0]=='<'

def get_tag(line):
    index = line.find('>')
    return line[1:index]

def get_content(line):
    index = line.find('>')
    return line[index+1:]

def split_sentence_hkib(article):
    return kkma.sentences(article)


def ReadHkib(path):
    lines = codecs.open(path, "r", "euc-kr", errors="replace").read().splitlines()

    f_in_text = False
    text_read = ""
    articles = []
    for index, line in enumerate(lines):
        if len(line) ==0 :
            f_in_text = False
            if len(text_read) > 0 :
                articles.append(text_read)
                text_read = ""
        elif contain_tag(line):
            tag = get_tag(line)
            if tag == "TEXT":
                f_in_text = True
                text_read += get_content(line)
            elif tag == "DOCID":
                f_in_text = False
                if len(text_read) > 0 :
                    articles.append(text_read)
                    text_read = ""
            else:
                if f_in_text:
                    raise Exception("Not Expected!!")
        else: # not tag
            if f_in_text:
                content = line[3:]
                if line[:3] != "   ":
                    raise Exception("At line {} : First three character must be space : '{}'".format(index,"mmm"))
                text_read += content

    print "total of {} articles read.".format(len(articles))
    return articles


kkma_count = 0

import re
def split_sentence(text):
    sentences = re.split(';|!|\\. |\\?', text)
    r = [s.strip() for s in sentences]
    return r


    return kkma.sentences(text)


if __name__ == "__main__":
    path_folder = """C:\work\Data\외부 데이터\HANTEC-2.0\DATA\hkib94"""

    all_article = []
    filenames = ["hkib94.001","hkib94.002","hkib94.003","hkib94.004","hkib94.005"]
    for filename in filenames:
        articles = ReadHkib(path_folder + "\\" + filename)
        all_article += articles

    print "Total articles : {}".format(len(all_article))

    all_sentence = flatten([split_sentence(article) for article in all_article])
    print "Total sentence : {}".format(len(all_sentence))

    with codecs.open("hkib_sentence.txt","w", encoding="euc-kr", errors="replace") as f:
        for s in all_sentence:
            f.write(s + "\n")




