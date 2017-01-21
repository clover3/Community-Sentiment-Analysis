
import random
from clover_lib import *

IDX_RATING = 0
IDX_TITLE = 1
IDX_CONTENT = 2
IDX_AUTHOR_ID = 3
IDX_ARTICLE_ID = 5
IDX_THREAD_ID = 6
IDX_TEXT_TYPE = 9
IDX_TOKENS = 10

IDX_R_THREAD = 1
IDX_R_ARTICLE = 2
IDX_R_KEYWORD = 3
IDX_R_LABEL = 4


def get_article_dic(articles):
    article_dic = dict()
    for article in articles:
        article_id = article[IDX_ARTICLE_ID]
        thread_id = article[IDX_THREAD_ID]
        article_dic[(article_id, thread_id)] = article
    return article_dic


def parse_token(articles):
    result = []
    for i,article in enumerate(articles):
        try:
            tokens = article[IDX_TOKENS].split('/')
            n_article = article[0:IDX_TOKENS] + [tokens]
            result.append(n_article)
        except Exception as e:
            print(i,(e))
    return result


def get_content(article):
    if len(article[IDX_TITLE]) > 0:
        content = article[IDX_TITLE] + ". " + article[IDX_CONTENT]
    else:
        content = article[IDX_CONTENT]
    return content


def extract_uanimous(path):
    IDX_L_CORPUS_ID = 0
    IDX_L_LABEL = 1
    labels = load_csv(path)

    filtered = set()
    label_index = dict()
    for label in labels:
        corpus_id = label[IDX_L_CORPUS_ID]
        if corpus_id in label_index:
            pre_label = label_index[corpus_id]
            if pre_label != label[IDX_L_LABEL]:
                filtered.add(corpus_id)
        else :
            label_index[corpus_id] = label[IDX_L_LABEL]

    new_data = []
    for label in labels:
        corpus_id = label[IDX_L_CORPUS_ID]
        if corpus_id not in filtered:
            new_data.append(label)

    save_csv_utf(new_data, "unanimous_label.csv")



def load_label(label_path):
    labels = load_csv(label_path)
    labels = [label for label in labels if
              label[IDX_R_LABEL] == '1' or label[IDX_R_LABEL] == '3' or label[IDX_R_LABEL] == '2']
    labels_pos = [label for label in labels if label[IDX_R_LABEL] == '1']
    labels_neu = [label for label in labels if label[IDX_R_LABEL] == '2']
    labels_neg = [label for label in labels if label[IDX_R_LABEL] == '3']

    n_pos =  len([label for label in labels if label[IDX_R_LABEL] == '1'])
    n_neu =  len([label for label in labels if label[IDX_R_LABEL] == '2'])
    n_neg = len([label for label in labels if label[IDX_R_LABEL] == '3'])
    print("  Corpus size : positive={} negative={} neutral={}...".format(n_pos, n_neg, n_neu) )

    labels = labels_pos + labels_neu + labels_neg
    random.shuffle(labels)
    return labels


def load_label_join(simple_label_path, corpus_path):
    simple_labels = load_csv(simple_label_path)
    corpus = load_csv(corpus_path)

    corpus_dic = dict()
    for c in corpus:
        corpus_dic[c[0]] = c

    label_dic = dict()
    for simple_label in simple_labels:
        label_dic[simple_label[0]] = simple_label[1]

    result = []
    for corpus_id in label_dic:
        c = corpus_dic[corpus_id]


        article_id = c[1]
        thread_id = c[2]
        keyword = c[4]
        label = label_dic[corpus_id]

        entry = [corpus_id, thread_id, article_id, keyword, label]
        result.append(entry)

    save_csv(result, "..\\input\\unanimous_label.csv")


load_label_join("simple_unanimous_label.csv", "..\\input\\corpus.csv")
