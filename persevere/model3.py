
import pandas

data=pandas.read_csv('./data/atec_nlp_sim_train.csv',sep='\t',header=None,names=['tid','text1','text2','label'])
input=data.text1

import jieba
jieba.load_userdict("./data/newdict.txt")

contexts=[]
for line in input:
    contexts.append(list(jieba.cut(line)))

from gensim.summarization import bm25
bm25 = bm25.BM25(contexts)
average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

def similarity(query, size=10):
    scores = bm25.get_scores(query, average_idf)
    scores_sort = sorted(list(enumerate(scores)),key=lambda item: item[1], reverse=True)
    return scores_sort[:size]

input_text='花呗透支了为什么不可以继续用了'
my_text=list(jieba.cut(input_text))
scores_sort=similarity(my_text,size=5)
for i ,j in scores_sort:
    print(''.join(contexts[i]))