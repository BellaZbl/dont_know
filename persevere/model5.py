from gensim import models
from gensim.similarities import WmdSimilarity


class WmdUtil(object):

    def __init__(self, word2vec_path):
        self.word2vec =models.word2vec.Word2Vec.load(word2vec_path)
        print ("Load word2bec model done.")


    def similarity(self, query, docs, size=10):
        wmd_inst = WmdSimilarity(docs, self.word2vec,num_best=size, normalize_w2v_and_replace=False)
        sims = wmd_inst[query]
        return sims

mymodel=WmdUtil('w2v.model')


import pandas

data=pandas.read_csv('./data/atec_nlp_sim_train.csv',sep='\t',header=None,names=['tid','text1','text2','label'])
input=data.text1

import jieba
jieba.load_userdict("./data/newdict.txt")

input_text='花呗透支了为什么不可以继续用了'
my_text=list(jieba.cut(input_text))

contexts=[]
for line in input:
    contexts.append(list(jieba.cut(line)))

sims=mymodel.similarity(my_text,contexts)
print(sims)