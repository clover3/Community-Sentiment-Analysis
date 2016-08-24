from clover_lib import *
import itertools
from konlpy.tag import Kkma


def remove_substring(word_list):
    word_list.sort(key=len)
    unique_words = []
    for index in range(len(word_list)):
        fAdd = True
        word = word_list[index]
        for word2 in word_list[index+1:]:
            if word in word2 :
                fAdd = False
                break
        if fAdd:
            unique_words.append(word)

    return unique_words


def match_all(content, lst):
    words = []
    for item in lst:
        if item.lower() in content.lower():
            words.append(item)

    return remove_substring(words)


def contain_any(content, lst):
    for item in lst:
        if item in content:
            print(item)
            return True
    return False


def content_contain_any(text, lst):
    content = text[IDX_TITLE] + " " + text[IDX_CONTENT]
    return contain_any(content, lst)


def thread_with_keyword(data):
    s = set()
    for text in data:
        thread_id = text[6]
        if thread_id not in s and content_contain_any(text, keywords):
            s.add(thread_id)
    return s


def filter_with_threads(data, threads):
    return list(filter(lambda text: text[6] in threads, data))


def filter_article(data, given_filter):
    return list(filter(given_filter, data))

    #filtered_data = filter_with_threads(data, thread_with_keyword(data) )


def get_filtered_bobaedream():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae"]
    car_keyword = load_list("input\\car_keyword.txt")
    keywords = car_keyword

    def article_with_car_keyword(article):
        return content_contain_any(article, car_keyword)

    data = load_csv_euc_kr("input\\bobae_car.csv")

    filtered_data = filter_article(data, article_with_car_keyword)

    save_csv_euc_kr(filtered_data,"input\\bobae_articles_car.csv")
    print("Total article :" + str(len(data)))
    print("Filtered article :" + str(len(filtered_data)))


def analyze_bobaedream():
    keyword_hyundai = ["현대","현기","흉기","기아","hyundae","현기차","현대차"]
    car_keyword = load_list("input\\car_keyword.txt")
    keywords = car_keyword + keyword_hyundai
    kkma = Kkma()
    data = load_csv_euc_kr("input\\bobae_car.csv")

    for article in data[:1000]:
        content = article[IDX_TITLE] + "\n" + article[IDX_CONTENT]
        #sentences = kkma.sentences(content)
        #for text in sentences:
        text = content
        words = match_all(text, keywords)
        if len(words) > 0 :
            found_words = ",".join(words)
            print("[{}] : {}".format(found_words, text))
            print("----------------------------------")



analyze_bobaedream()