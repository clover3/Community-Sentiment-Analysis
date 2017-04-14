from clover_lib import *
from KoreanNLP import *
import random
import re
import pickle
from idx2word import Idx2Word
from entity_dict import EntityDict
from konlpy import tag

# Sentence-to-Entity Affinity Learner

def remove_keyword(text, matched_keyword):
    reg = r"(^|\s)" + matched_keyword + ".*?($|\s)"
    removed = re.sub(reg, " ", text)
    return removed

def S2ECorpus(raw_corpus_path, entity_dict, save_path, idx2word):
# 1. Load all documents
    doc = load_csv_euc_kr(raw_corpus_path)

    # [ int->List[String] ]
    e2s_dict = dict()

    def push(entity_id, sent):
        if entity_id not in e2s_dict:
            e2s_dict[entity_id] = list()
        e2s_dict[entity_id].append(sent)

    all_sentence = []
    # 2. Generate Entity - Sentence Dictionary
    #    Remove entity from the sentence
    print("Iterating docs...")
    bar = ListProgress(doc)
    for article in doc:
        bar.step()
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        for sentence in split_sentence(content):
            matching_entitys = entity_dict.extract_from(sentence)
            for entity in matching_entitys:
                sent_removed = remove_keyword(sentence, entity)
                entity_id = entity_dict.get_group(entity)
                push(entity_id, sent_removed)
                all_sentence.append(sent_removed)

            if len(matching_entitys) == 0:
                push(0, sentence)
                all_sentence.append(sentence)

    def pick_random_sentence():
        return random.choice(all_sentence)

    twitter = tag.Twitter()

    def to_index(str):
        poses = twitter.pos(str, True, True)
        res = [idx2word.word2idx(pos[0]) for pos in poses]
        return res
    # Corpus Gen
    #
    # 3. Sentence / Entity / Label
    print("Generating corpus...")
    corpus = []
    bar = ListProgress(e2s_dict.keys())
    for key in e2s_dict.keys():
        bar.step()
        if key == 0:
            continue
        related_sentences = e2s_dict[key]
        # Positive Corpus
        for sent in related_sentences:
            corpus.append((key, to_index(sent), 1))
           # print((entity_dict.get_name(key),sent,1))
        # Negative Corpus
        for i in range(len(related_sentences)):
            sent = pick_random_sentence()
            while sent in related_sentences:
                sent = pick_random_sentence()
            corpus.append((key, to_index(sent), 0))
            #print((entity_dict.get_name(key), sent, 0))

    print("Saving pickle...")
    pickle.dump(corpus, open(save_path,"wb"))

# Training Phase



if __name__ == "__main__":
    idx2word = Idx2Word("data\\idx2word")
    entity_dict = EntityDict("..\\input\\EntityDict.txt")
    S2ECorpus("..\\input\\bobae_car_euc.csv", entity_dict, "data\\E2S.p", idx2word)