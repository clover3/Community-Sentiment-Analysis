# -*- coding: utf-8 -*-
from clover_lib import *
from konlpy.tag import Kkma
from KoreanNLP import *
import re
import random


def make_positive_corpus():
    keywords = ["말리부", "쏘나타", "그랜져", "그랜저"]
    data = load_csv("..\\input\\bobae_car_tkn_twitter.csv")
    corpus = []
    kkma = Kkma()

    def get_keyword_context(text, keyword, window_size):
        keyword_locations = []

        tokens = text.split()
        for i, item in enumerate(tokens):
            if keyword in item:
                keyword_locations.append(i)

        entrys = []
        for location in keyword_locations:
            prev = tokens[location - window_size : location]
            middle = ["[" + tokens[location] + "]"]
            next = tokens[location+1 : location + window_size +1]

            output = " ".join(prev + middle + next)
            entrys.append(output)
        return entrys

    def remove_keyword(text, matched_keyword):
        reg = r"(^|\s)" + matched_keyword + ".*?($|\s)"
        removed = re.sub(reg, " ", text)
        return removed

    corpus_index = 0
    for article in data:
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        matched_keyword = contain_any(content, keywords)
        if matched_keyword:
            sentences = split_sentence(content)
            for text in sentences:
                if matched_keyword in text:
                    contexts = get_keyword_context(text, matched_keyword, 5)

                    for text in contexts:
                        entry = remove_newline(text)
                        corpus.append(entry)
                        corpus_index += 1
    save_list(corpus, "..\\input\\car_context_view.txt")



def make_negative_corpus():
    keywords = ["말리부", "쏘나타", "그랜져", "그랜저"]
    data = load_csv("..\\input\\bobae_car_tkn_twitter.csv")
    corpus = []


    def get_random_context(text, window_size):
        tokens = text.split()
        if len(tokens) > 3 :
            location = random.randrange(0,len(tokens))
            prev = tokens[location - window_size: location]
            next = tokens[location+1 : location + window_size + 1]

            output = " ".join(prev + next)
        else:
            output = ""
        return output


    corpus_index = 0
    for article in data:
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        if not contain_any(content, keywords):
            sentences = split_sentence(content)
            for sentence in sentences:
                context = get_random_context(sentence, 5)
                if context :
                    entry = remove_newline(context)
                    corpus.append(entry)
                    corpus_index += 1
        if corpus_index > 100000:
            break
    save_list(corpus, "..\\input\\not_car_context.txt")


if __name__ == '__main__':
    make_positive_corpus()
    #make_negative_corpus()