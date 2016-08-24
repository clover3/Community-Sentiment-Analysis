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

