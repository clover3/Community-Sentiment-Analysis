
from clover_lib import *

from konlpy.tag import Kkma
from konlpy.tag import Twitter


def tokenize_all(articles):
    tagger = Kkma()
    result = []
    count = 0
    for article in articles:
        count += 1
        if count % 100 == 0 :
            print(count)
        try:
            text = article[IDX_TITLE] + " " + article[IDX_CONTENT]
            tokens = list(map(lambda x: x[0], tagger.pos(text)))
            token_str = "/".join(tokens)
            n_article = article + [token_str]
            result.append(n_article)
        except Exception as e:
            print(e)
            print("Exception at article ", article[IDX_ARTICLE_ID])
    return result


def add_tokenzied(in_path, out_path):
    print("DEBUG : Tokenizing corpus...")
    data = load_csv_euc_kr(in_path)[0:2000]
    data = tokenize_all(data)
    save_csv_euc_kr(data, out_path)

#add_tokenzied("input\\clean_guldang.csv", "input\\clean_guldang_tkn.csv")
add_tokenzied("input\\bobae_car.csv", "input\\bobae_car_tkn.csv")