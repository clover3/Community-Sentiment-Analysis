
import codecs

from pydbscan import DBSCAN

def load_w2v(path):
    entries = codecs.open(path, "r", 'utf-8').readlines()[1:]
    def parse_line(entry):
        tokens = entry.split(" ")
        word = tokens[0]
        vectors = list(map(lambda x:float(x), tokens[1:]))
        return (word,vectors)

    parsed_entries = dict(list(map(parse_line, entries)))
    return parsed_entries


dic = load_w2v("input\\korean_word2vec_wv_50.txt")

data = dic.values()

db = DBSCAN()

db.init( 0.5, 5, 4 )
db.fit(data)

labels = db.get_labels()


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Node : %d' % len(dic))
print('Estimated number of clusters: %d' % n_clusters_)