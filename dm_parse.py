# -*- coding: utf-8 -*-

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
"""

from clover_lib import *
import itertools



def analyze_guldang():
    data = load_csv_euc_kr("input\\clean_guldang.csv")

    for text in data:
        if len(text) < 9 :
            print(text)


    count = 0
    for key, group in itertools.groupby(data, lambda x: x[6]):
        texts = list(group)

        users_raw = map(lambda x: x[3], texts)
        users = set(users_raw)

        for text in texts:
            content = text[2]
            for user in users:
                if user in content:
                    count += 1
                    print(content)

    print(count)

def reorder_bobaedream():
    data = load_csv_euc_kr("input\\bobae_national.csv")

    thread_dic = dict()

    for text in data:
        thread_id = text[6]
        if thread_id in thread_dic:
            l = thread_dic[thread_id]
            l.append(text)
        else:
            l = list()
            l.append(text)
            thread_dic[thread_id] = l

    def flatten(ll):
        result_list = list()
        for l in ll:
            for entry in l:
                result_list.append(entry)
        return result_list

    ordered_data = flatten(thread_dic.values())
    save_csv_euc_kr(ordered_data, "input\\bobae_national_ordered.csv")

def analyze_bobaedream():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae"]
    car_keyword = load_list("input\\car_keyword.txt")
    keywords = car_keyword

    data = load_csv_euc_kr("input\\bobae_national_ordered.csv")

    def contain_any(content, lst):
        for item in lst:
            if item in content:
                print(item)
                return True
        return False

    def content_contain_any(text, lst):
        content = text[IDX_TITLE] + " " + text[IDX_CONTENT]
        return contain_any(content, lst)



    def thread_with_keyword(data):
        s = set()
        for text in data:
            thread_id = text[6]
            if thread_id not in s and content_contain_any(text, keywords):
                s.add(thread_id)
        return s

    def filter_with_threads(data, threads):
        return list(filter(lambda text: text[6] in threads, data))

    def article_with_car_keyword(article):
        return content_contain_any(article, car_keyword)

    def filter_article(data, given_filter):
        return list(filter(given_filter, data))

    #filtered_data = filter_with_threads(data, thread_with_keyword(data) )

    filtered_data = filter_article(data, article_with_car_keyword)

    save_csv_euc_kr(filtered_data,"input\\bobae_articles_car.csv")
    print("Total article :" + str(len(data)))
    print("Filtered article :" + str(len(filtered_data)))


analyze_bobaedream()
