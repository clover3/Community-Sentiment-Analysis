

from clover_lib import *
import pickle

## load target pickle
EL_IDX_THREAD_ID = 1
EL_IDX_ARTICLE_ID = 2
EL_IDX_KEYWORD = 3
EL_IDX_LABEL = 4

relation = pickle.load(open("targetDict.p", "rb"))

data = load_csv("..\\input\\corpus_samba.csv")

def average(l):
    sum = 0
    for item in l:
        sum += item

    return float(sum) / len(l)

print len(data)
def get_tid(article):
    return int(article[EL_IDX_THREAD_ID]), int(article[EL_IDX_ARTICLE_ID])

def get_tlid(article):
    return int(article[EL_IDX_THREAD_ID]), int(article[EL_IDX_ARTICLE_ID]), article[EL_IDX_KEYWORD]


def get_article_dic(articles):
    article_dic = dict()
    for article in articles:
        article_dic[get_tlid(article)] = article
    return article_dic

article_dict = get_article_dic(data)

yes = 0
no =0

pp = FailCounter()
nn = FailCounter()
p00 = FailCounter()


for article in data:
    if get_tid(article) in relation:
        pre_thread_id, pre_article_id = relation[get_tid(article)]
        pre_keyword = article[EL_IDX_KEYWORD]
        pre_id = (pre_thread_id, pre_article_id, pre_keyword)
        if pre_article_id == article[EL_IDX_ARTICLE_ID]:
            continue

        if pre_id in article_dict:
            pre_article = article_dict[pre_id]
            #print("label pre : {} , label next {}".format(pre_article[EL_IDX_LABEL], article[EL_IDX_LABEL]))
            if pre_article[EL_IDX_LABEL] == article[EL_IDX_LABEL] :
                yes += 1
            else:
                no +=1

            if pre_article[EL_IDX_LABEL] == '1':
                if article[EL_IDX_LABEL] == '1':
                    pp.suc()
                else:
                    pp.fail()

            if pre_article[EL_IDX_LABEL] == '3':
                if article[EL_IDX_LABEL] == '3':
                    nn.suc()
                else:
                    nn.fail()

            if pre_article[EL_IDX_LABEL] == '2':
                if article[EL_IDX_LABEL] == '2':
                    p00.suc()
                else:
                    p00.fail()
        else:
            None

print("Yes/No ={}/{}".format(yes,no))
print("Yes/No ={}".format(float(yes)/(yes+no)))
print("P-P = {}".format(pp.precision()))
print("N-N = {}".format(nn.precision()))
print("0-0 = {}".format(p00.precision()))

group_dict = dict()

for article in data:
    id = article[EL_IDX_THREAD_ID], article[EL_IDX_KEYWORD]
    if id not in group_dict:
        group_dict[id] = list()

    group_dict[id].append(article)

portion_list = []

for key in group_dict.keys():
    group = group_dict[key]
    if len(group) > 3 :
        cc = CaseCounter()
        for article in group :
            cc.add_count(article[EL_IDX_LABEL])
        ml = cc.majority_label()
        if ml != '2':
            portion = float(cc.majority_size()) / cc.total()
            portion_list.append(portion)
            print ml, portion

print("average majority : {}".format(average(portion_list)))


## load corpus