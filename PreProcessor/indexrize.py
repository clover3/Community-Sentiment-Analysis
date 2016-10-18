#### This file converts words to index 

from clover_lib import *


def indexrize(path, out_path, fSentenceWise, idx2cluster = None):
    articles = load_csv(path)
    sentences = []
    print articles[0][IDX_TOKENS]

    if fSentenceWise:
        sentences = parse_sentence_token(articles)
    else:
        articles = parse_token(articles)
        sentences = [article[IDX_TOKENS] for article in articles]
    ss = flatten(sentences)

    voca = set(flatten(sentences))

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

    indexed_articles = [sentence2intList(sentence) for sentence in sentences]

    output_array(out_path, indexed_articles)

    idx2word = {v: k for k, v in word2idx.iteritems()}
    return idx2word

def convert2viwable(path, idx2word):
    f = open(path, "r")


def output_array(path, indexed_article):
    fp= open(path, "w")

    for article in indexed_article:
        for token in article:
            fp.write("{} ".format(token))
        fp.write("\n")
    fp.close()

def load_cluster(path):
    data = open(path, "r").readlines()
    idx2cluster = dict()
    for line in data:
        tokens = line.strip().split(" ")
        cluster = tokens[0]
        items = tokens[1:]
        for item in items:
            idx2cluster[int(item)] = int(cluster)
    print idx2cluster
    return idx2cluster

if __name__ == '__main__':
    #indexrize("..\\input\\bobae_car_tkn_twitter.csv", False)
    idx2cluster = load_cluster("..\\input\\cluster_index.txt")
    indexrize("..\\input\\babae_car_tokens_sentence_wise.csv","..\\input\\bobae_index_clustered", True,idx2cluster)