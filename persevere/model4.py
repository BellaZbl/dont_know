from gensim import models


import pandas

data1=pandas.read_csv('./data/atec_nlp_sim_train.csv',sep='\t',header=None,names=['tid','text1','text2','label'])
input1=data1.text1+data1.text2


data2=pandas.read_csv('./data/atec_nlp_sim_train_add.csv',sep='\t',header=None,names=['tid','text1','text2','label'])
input2=data2.text1+data2.text2


input=pandas.concat([input1,input2],axis=0)


import jieba
jieba.load_userdict("./data/newdict.txt")

rfd=[]
for line in input:
    outpu=list(jieba.cut(line))
    outpu=' '.join(outpu)
    rfd.append(outpu)

def _input_streaming():
    for line in rfd:
        word_seg = line.strip("\n").split()
        yield word_seg


def train_word2vec_model():
    vec_size = 100
    win_size = 7
    corpus_ = [s.split() for s in rfd]
    # begin to train
    print("begin to train model...")
    w2v_model = models.word2vec.Word2Vec(corpus_,
                                         size = vec_size,
                                         window = win_size,
                                         min_count = 5,
                                         workers = 4,
                                         sg = 0,
                                         negative = 10,
                                         iter =300)
    #w2v_model.train(_input_streaming(),total_examples=len(corpus_), epochs=w2v_model.iter)
    print(w2v_model.most_similar(positive='花呗'))
    print(w2v_model.most_similar(positive='付款'))
    print(w2v_model.similarity('还款','付款'))
    print(w2v_model.similarity('客户','付款'))


train_word2vec_model()