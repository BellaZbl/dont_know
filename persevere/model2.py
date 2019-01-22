import pandas

data=pandas.read_csv('./data/atec_nlp_sim_train.csv',sep='\t',header=None,names=['tid','text1','text2','label'])
input=data.text1

import jieba
jieba.load_userdict("./data/newdict.txt")

contexts=[]
for line in input:
    contexts.append(list(jieba.cut(line)))

from gensim import corpora
dct=corpora.Dictionary(contexts)
low_freq_ids =[id_ for id_, freq in dct.dfs.items() if freq<3]
high_freq_ids=[id_ for id_, freq in dct.dfs.items() if freq>10000]
freq_ids=low_freq_ids+high_freq_ids
dct.filter_tokens(freq_ids)
dct.compactify()
corpus = [dct.doc2bow(s) for s in contexts]

from gensim import models
tfidf_model = models.TfidfModel(corpus)
corpus_mm=tfidf_model[corpus]

from gensim import similarities
index = similarities.MatrixSimilarity(corpus_mm, num_features=len(dct))

def text2vec(text):
    bow = dct.doc2bow(text)
    return tfidf_model[bow]

input_text='花呗透支了为什么不可以继续用了'
my_text=list(jieba.cut(input_text))
vec =text2vec(my_text)
sims = index[vec]
sim_sort = sorted(list(enumerate(sims)),key=lambda item: item[1], reverse=True)

docs = [contexts[id_] for id_, score in sim_sort[:5]]

for i in docs[:10]:
    print(''.join(i))