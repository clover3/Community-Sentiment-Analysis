from clover_lib import *


class Idx2Word:
    # 0 : Null padding
    # 1 : Unknown word
    def __init__(self, path):
        data = codecs.open(path, "r", encoding="cp949").readlines()
        self.idx2word_ = dict()
        self.word2idx_ = dict()
        self.voca_size = 5

        self.trial = 0
        self.not_found = 0

        for line in data:
            idx = int(line.split()[0])
            word = (line.split())[1].strip()
            assert type(word) == type("word")
            self.idx2word_[idx] = word
            self.word2idx_[word] = idx

            if idx >= self.voca_size :
                self.voca_size = idx+1

    def get_voca(self):
        return self.word2idx_.keys()

    def idx2word(self, idx):
        if idx in self.idx2word_:
            return self.idx2word_[idx]
        if idx == 0 :
            return "empty"
        else:
            return "Unknown"

    def word2idx(self, word):
        self.trial += 1
        if word in self.word2idx_:
            return self.word2idx_[word]
        else:
            self.not_found += 1
            return 1
