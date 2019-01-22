import pandas as pd
from tensorflow.contrib import learn
import numpy as np

texts1=pd.read_csv('./data/atec_nlp_sim_train.csv',delimiter='\t',header=None,names=['row_id','seq1','seq2','lable'])
texts2=pd.read_csv('./data/atec_nlp_sim_train_add.csv',delimiter='\t',header=None,names=['row_id','seq1','seq2','lable'])
texts=pd.concat([texts1,texts2],axis=0)

stopwords=['？','，','。']

sequences1=texts.seq1
sequences1=[[word.strip() for word in line.encode('utf-8').decode('utf-8-sig').strip() if word not in stopwords] for line in sequences1]
sequences1=[' '.join(line) for line in sequences1]

sequences2=texts.seq2
sequences2=[[word.strip() for word in line.encode('utf-8').decode('utf-8-sig').strip() if word not in stopwords] for line in sequences2]
sequences2=[' '.join(line) for line in sequences2]

labels=np.array(texts.lable)

sequences=sequences1+sequences2

vocab_processor=learn.preprocessing.VocabularyProcessor(30)
vocab_processor.fit(sequences)

seq1=np.array(list(vocab_processor.transform(sequences1)))
seq2=np.array(list(vocab_processor.transform(sequences2)))

vocab_processor.save('vocab_processor.model')

import pickle

print(type(seq1))
file1=open('./data/seq1.pkl','wb')
pickle.dump(seq1,file1)

file2=open('./data/seq2.pkl','wb')
pickle.dump(seq2,file2)

file3=open('./data/lables.pkl','wb')
pickle.dump(labels,file3)

