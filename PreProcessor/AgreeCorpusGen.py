from clover_lib import *
import sqlite3

def load_sql(sqlpath,label):
    conn = sqlite3.connect(sqlpath)
    c = conn.cursor()

    query_str = "select content from AgreeReports,AgreeCorpus where label=\"{}\" and AgreeReports.corpus_id=AgreeCorpus.id;".format(label);

    result = []
    for row in c.execute(query_str):
        result.append(unicode2str(row[0])+"\n")
    return result

def save_list(data, path):
    fp = open(path, "w")
    fp.writelines(data)
    fp.close()


def writefile(data_neu, data_pos, data_neg):
    path_prefix = "..\\input\\agree_"
    neu_path = path_prefix + "0.csv"
    pos_path = path_prefix + "1.csv"
    neg_path = path_prefix + "2.csv"

    save_list(data_neu, neu_path)
    save_list(data_pos, pos_path)
    save_list(data_neg, neg_path)




if __name__ == '__main__':
    data_pos = load_sql("d:\\data\\rp.db", 1)
    data_neg = load_sql("d:\\data\\rp.db", 2)
    data_neu = load_sql("d:\\data\\rp.db", 3)
    writefile(data_neu, data_pos, data_neg)

