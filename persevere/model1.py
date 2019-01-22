import jieba

file='./data/qa_text.txt'

contexts=[]
responses=[]
with open(file) as f:
    for line in f.readlines():
        query,answer,_=line.replace('"','').split(',')
        contexts.append(list(jieba.cut(query)))
        responses.append([answer])

print(len(contexts))

from gensim import corpora
dct=corpora.Dictionary(contexts)
low_freq_ids =[id_ for id_, freq in dct.dfs.items() if freq<3]
high_freq_ids=[id_ for id_, freq in dct.dfs.items() if freq>100]
freq_ids=low_freq_ids+high_freq_ids
dct.filter_tokens(freq_ids)
dct.compactify()
corpus = [dct.doc2bow(s) for s in contexts]


from gensim import models
tfidf_model = models.TfidfModel(corpus)
corpus_mm=tfidf_model[corpus]

from gensim import similarities
index = similarities.Similarity(None,corpus_mm, num_features=len(dct))


def text2vec(text):
    bow = dct.doc2bow(text)
    return tfidf_model[bow]

input_text='北京是哪个国家的城市'
my_text=list(jieba.cut(input_text))
vec =text2vec(my_text)
sims = index[vec]
sim_sort = sorted(list(enumerate(sims)),key=lambda item: item[1], reverse=True)

docs = [contexts[id_] for id_, score in sim_sort[:5]]
answers = [responses[id_] for id_, score in sim_sort[:5]]

for i,j in zip(docs,answers):
    print(''.join(i),j[0])


