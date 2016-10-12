#### This file converts words to index 

from clover_lib import *

articles = load_csv("..\\input\\bobae_car_tkn_twitter.csv")
print articles[0]
print articles[0][IDX_TOKENS]
articles = parse_token(articles)
print articles[0][IDX_TOKENS]
voca = set(flatten([article[IDX_TOKENS] for article in articles]))

word2idx = dict()

index = 5
for word in voca:
    word2idx[word]= index
    index += 1

indexed_articles = []
for article in articles:
    words = article[IDX_TOKENS]
    indexed_article = [word2idx[word] for word in words]
    indexed_articles.append(indexed_article)

fout= open("bobae_as_index", "w")

for article in indexed_articles:
    for token in article:
        fout.write("{} ".format(token))
    fout.write("\n")
fout.close()


