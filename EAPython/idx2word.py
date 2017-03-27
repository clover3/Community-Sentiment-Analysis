from clover_lib import *


class Idx2Word:
    # 0 : Null padding
    # 1 : Unknown word
    def __init__(self, path):
        data = codecs.open(path, "r", encoding="cp949").readlines()
        self.idx2word_ = dict()
        self.word2idx_ = dict()
        self.voca_size = 5

        for line in data:
            idx = int(line.split()[0])
            word = (line.split())[1].strip()
            assert type(word) == type("word")
            self.idx2word_[idx] = word
            self.word2idx_[word] = idx

            if idx >= self.voca_size :
                self.voca_size = idx+1


    def idx2word(self, idx):
        if idx in self.idx2word_:
            return self.idx2word_[idx]
        else:
            return "Unknown"

    def word2idx(self, word):
        if word in self.word2idx_:
            return self.word2idx_[word]
        else:
            return 1
