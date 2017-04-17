from clover_lib import *
from KoreanNLP import *
import random
import pickle
from idx2word import Idx2Word
from entity_dict import EntityDict
from konlpy import tag

def gen_corpus():
    idx2word = Idx2Word("data\\idx2word")
    entity_dict = EntityDict("..\\input\\EntityDict.txt")
    target_size = 10000
    raw_corpus = load_csv_euc_kr("..\\input\\bobae_car_euc.csv")[:target_size * 10 ]

    target_dictionary = pickle.load(open("data\\targetDict.p","rb"))



    def get_thread_article_id(post):
        return post[IDX_THREAD_ID], post[IDX_ARTICLE_ID]
    def get_content(article):
        return article[IDX_TITLE] + "\n" + article[IDX_CONTENT]

    sent_dict = dict()
    for sent in raw_corpus:
        sent_dict[get_thread_article_id(sent)] = sent


    def get_prev_sent(article):
        id = get_thread_article_id(article)
        tid, aid = id
        if id in target_dictionary and tid != aid:
            pre_id = target_dictionary[id]
            pre_article = sent_dict[pre_id]
            return split_sentence(get_content(pre_article))[-1]
        else:
            return ""

    def gen(article):
        result = []
        content = get_content(article)
        prev_sent = get_prev_sent(article)

        sentences = split_sentence(content)
        for i, sentence in enumerate(sentences):
            if i == 0:
                if prev_sent:
                    result.append((prev_sent, sentence))
            else:
                result.append((sentences[i-1], sentence))
        return result

    pos_data = flatten([gen(article) for article in raw_corpus])[:target_size]

    all_sentence = flatten([split_sentence(get_content(article)) for article in raw_corpus])

    neg_data = []
    for i in range(target_size):
        a = random.choice(all_sentence)
        b = random.choice(all_sentence)
        neg_data.append((a,b))

    twitter = tag.Twitter()

    def to_index(str):
        poses = twitter.pos(str, True, True)
        res = [idx2word.word2idx(pos[0]) for pos in poses]
        return res

    print("Pos : {} Neg : {}".format(len(pos_data), len(neg_data)))

    pos_data_idxed = []
    for p in pos_data:
        a,b = p
        token = (to_index(a), to_index(b), 1)
        pos_data_idxed.append(token)

    neg_data_idxed = []
    for p in neg_data:
        a,b = p
        token = (to_index(a), to_index(b), 1)
        neg_data_idxed.append(token)


    pickle.dump((pos_data_idxed, neg_data_idxed), open("S2Q.p", "wb"))


gen_corpus()









