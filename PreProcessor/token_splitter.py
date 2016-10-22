
from clover_lib import *


def output_array(path, indexed_article):
    fp= open(path, "w")

    for article in indexed_article:
        for token in article:
            fp.write("{}\t".format(token))
        fp.write("\n")
    fp.close()

def splitter(in_path, out_path):
    articles = load_csv(in_path)
    sentences = []
    print articles[0][IDX_TOKENS]

    sentences = parse_sentence_token(articles)


    print "Total of {} sentences".format(len(sentences))
    output_array(out_path, sentences)



if __name__ == '__main__':
    splitter("..\\input\\babae_car_tokens_sentence_wise.csv", "..\\input\\bobae_tokened_corpus.txt")