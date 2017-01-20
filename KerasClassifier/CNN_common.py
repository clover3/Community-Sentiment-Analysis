
import numpy


def tokenize(list_str):
    from konlpy.tag import Twitter
    from konlpy.tag import Hannanum
    lib = Twitter()

    arr = []
    for sentence in list_str:
        if type(sentence) == type("str"):
            sentence = unicode(sentence,'utf-8')

        poses = lib.pos(sentence)
        tokens = map(lambda x:x[0].encode('utf8'), poses)
        arr.append(tokens)
    return arr



class SentenceSplitter:
    def __init__(self):
        from konlpy.tag import Kkma
        self.kkma = Kkma()

    def split(self, str):
        input = unicode(str, "utf-8")
        unicode_s = self.kkma.sentences(input)
        r =  [u.encode("utf-8","ignore") for u in unicode_s]
        return r




def load_vec(fname, vocab, binary = True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("  Loading word2vec...")
    #w2v_cache = "cache\\w2v"
    #if os.path.isfile(w2v_cache):
    #    return cPickle.load(open(w2v_cache,"rb"))

    mode = ("rb" if binary else "r")
    word_vecs = {}
    with open(fname, mode) as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * layer1_size

        def getline():
            if binary:
                return numpy.fromstring(f.read(binary_len), dtype='float32')
            else:
                return numpy.array(f.readline().split(), dtype='float32')

        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = getline()
            else:
                getline()
    print("  Loaded word2vec...")
#    cPickle.dump(word_vecs, open(w2v_cache, "wb"))
    return word_vecs


def build_index(voca, w2v, k):
    print("building index..")
    predefined_word = 5
    index = predefined_word
    word2idx = dict()
    idx2vect = numpy.zeros(shape=(len(voca) + predefined_word, k), dtype='float32')

    for i in range(predefined_word):
        #idx2vect[i] = numpy.zeros(k, dtype='float32')
        idx2vect[i] = numpy.random.uniform(-0.25,0.25,k)

    f = open("missing_w2v.txt", "w")
    if w2v is not None:
        for word in w2v.keys():
            word2idx[word] = index
            idx2vect[index] = w2v[word]
            index += 1

    match_count = index

    for word in voca:
        if word not in word2idx:
            f.write(word + "\n")
            word2idx[word] = index
            idx2vect[index] = numpy.zeros(k, dtype='float32')
            index += 1
    f.close()

    print("w2v {} of {} matched".format(match_count, index))
    return word2idx, idx2vect
