# -*- coding: euc-kr -*-
#### This file converts words to index

from clover_lib import *
from konlpy.tag import Twitter
import jpype


def init_jvm():
    print jpype.getDefaultJVMPath()
    jpype.startJVM(jpype.getDefaultJVMPath())

def make_word2idx(s):
    word2idx = dict()
    index = 5
    for word in s:
        word2idx[word] = index
        index += 1
    return word2idx

def indexrize_clover_sentence(path, out_path):
    articles = load_csv(path)
    sentences = []

    all_sentence = parse_sentence_token(articles)

    filter( lambda x: x[IDX_TEXT_TYPE]=="0", articles)

    head_articles = get_sentence_token(articles)
    head_sentences = [article[0] for article in head_articles]

    word2idx = make_word2idx(set(flatten(all_sentence)))

    def sentence2intList(sentence):
        return [word2idx[word] for word in sentence]

    indexed_articles = [sentence2intList(sentence) for sentence in head_sentences]

    output_array(out_path, indexed_articles)

    idx2word = {v: k for k, v in word2idx.iteritems()}
    return idx2word, word2idx



def indexrize(path, out_path, fSentenceWise, idx2cluster = None):
    articles = load_csv(path)
    sentences = []
    print articles[0][IDX_TOKENS]

    voca = set(flatten(parse_sentence_token(articles)))
    word2idx = dict()
    index = 5
    for word in voca:
        word2idx[word] = index
        index += 1

    if fSentenceWise:
        sentences = parse_sentence_token(articles)
    else:
        articles = parse_token(articles)
        filter( lambda x: x[IDX_TEXT_TYPE]=="0", articles)
        sentences = [article[IDX_TOKENS] for article in articles]



    def idx2group(idx):
        if idx2cluster is not None :
            if idx in idx2cluster:
                return idx2cluster[idx] + 1000000
        return idx

    def sentence2intList(sentence):
        return [idx2group(word2idx[word]) for word in sentence]

    if fSentenceWise:
        indexed_articles = [sentence2intList(sentence)  + ["||" + sentence] for sentence in sentences]
    else :
        indexed_articles = [sentence2intList(sentence) for sentence in sentences]

    output_array(out_path, indexed_articles)

    idx2word = {v: k for k, v in word2idx.iteritems()}
    return idx2word



def indexrize_k(path, out_path, idx2cluster = None):
    articles = load_csv_euc_kr(path)



    def get_type(x):
        try:
            type = x[IDX_TEXT_TYPE]
        except:
            print x
            return False

        return type

    filter( lambda x:get_type(x)=="0", articles)


    sentences = [article[IDX_TITLE] + "\n" + article[IDX_CONTENT] for article in articles]

    #init_jvm()
    from konlpy.tag import Kkma
    lib = Kkma()

    eccount = 0

    def getPos(str):
        try:
            r = lib.pos(unicode(str+" ", "utf-8"))
        except jpype._jexception.JavaException:
            r = []

        return r


    def string2tokens(str):
        return [token[0] for token in getPos(str)]

    tokens_list = [string2tokens(s) for s in sentences]

    voca = set(flatten(tokens_list))

    word2idx = dict()

    index = 5
    for word in voca:
        word2idx[word]= index
        index += 1


    def idx2group(idx):
        if idx2cluster is not None :
            if idx in idx2cluster:
                return idx2cluster[idx] + 1000000
        return idx

    def sentence2intList(sentence):
        return [idx2group(word2idx[word]) for word in sentence]

    indexed_articles = []

    try:
        for sentence in sentences:
            indexed_articles.append(sentence2intList(sentence))
    except:
        ""

    output_array(out_path, indexed_articles)

    idx2word = {v: k for k, v in word2idx.iteritems()}
    return idx2word


def indexrize_rawcorpus(path, out_path, word2idx):
    lines = codecs.open(path, "r", encoding="euc-kr", errors="replace").read().splitlines()

    tokenizer = Twitter()

    flog = codecs.open("temp", "w", encoding="euc-kr", errors="replace")
    list_indexs = []

    for i,line in enumerate(lines):
        poses = tokenizer.pos(line, True, True)
        tokens = map(lambda x: x[0], poses)
        tokens_filtered = filter(lambda x : x in word2idx, tokens)
        indexs = map(lambda x: word2idx[x], tokens_filtered)
        list_indexs.append(indexs)

        if i % 1000 == 0 :
            print i

    output_array(out_path, list_indexs)


def convert2viwable(path, outpath, idx2word):
    lines = codecs.open(path, "r", "cp949").readlines()
    fout = codecs.open(outpath, "w", "cp949")
    for line in lines:
        tokens = line.strip().split()
        outStr = ""
        for token in tokens:
            outStr += (idx2word[int(token)] + " ")
        fout.write(outStr + "\n")
    fout.close()

def load_cluster(path):
    data = open(path, "r").readlines()
    idx2cluster = dict()
    for line in data:
        tokens = line.strip().split(" ")
        cluster = tokens[0]
        items = tokens[1:]
        for item in items:
            idx2cluster[int(item)] = int(cluster)
    return idx2cluster

def print_idx2word(idx2word):
    f = open("idx2word","w")

    for entry in idx2word:
        f.write("{}\t{}\n".format(entry, idx2word[entry]))

    f.close()

def load_idx2word(path, cluster = None):
    data = codecs.open(path, "r", encoding="cp949").readlines()
    idx2word = dict()
    word2idx = dict()
    for line in data:
        idx = int(line.split()[0])
        word = (line.split())[1]
        idx2word[idx] = word
        word2idx[word] = idx
    if cluster is not None :
        for voca_id in cluster.keys() :
            if voca_id > 0 :
                cluster_id = cluster[voca_id]
                idx2word[cluster_id + 100000000] = idx2word[voca_id]

    return idx2word, word2idx


import code

if __name__ == '__main__':
    command = 1
    #indexrize("..\\input\\bobae_car_tkn_twitter.csv", False)
    #idx2cluster = load_cluster("..\\input\\cluster_index.txt")
    if command == 0:
        idx2word, word2idx = indexrize_clover_sentence("..\\input\\babae_car_tokens_sentence_wise.csv","..\\input\\bobae_index_only_article")
    if command == 1:

        idx2word, word2idx = load_idx2word("..\\input\\idx2word")
        path =""
        indexrize_rawcorpus("hkib_sentence.txt", "hkib_index.txt", word2idx)
    #idx2word = indexrize("..\\input\\babae_car_tokens_sentence_wise.csv","..\\input\\dummy", True, None)
    #print_idx2word(idx2word)

    if command == 2:
        cluster = load_cluster("..\\input\\cluster_ep200.txt")
        idx2word, word2idx= load_idx2word("..\\input\\idx2word", cluster)
        convert2viwable("..\\input\\L2_ep200.txt", "L2v.txt", idx2word )

    #indexrize_k("..\\input\\bobae_car_euc.csv","..\\input\\bobae_car_ktoken.csv", False)