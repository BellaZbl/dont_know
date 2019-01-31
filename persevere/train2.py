import tensorflow as tf
from tensorflow.contrib import learn
import pickle
import numpy as np
from persevere import model8

vocab_processor=learn.preprocessing.VocabularyProcessor.restore('vocab_processor.model')
n_vocab=len(vocab_processor.vocabulary_)

x1=pickle.load(open('./data/seq1.pkl','rb'))
x2=pickle.load(open('./data/seq2.pkl','rb'))
y=pickle.load(open('./data/lables.pkl','rb'))

shuffled=np.random.permutation(range(len(x1)))
x1=x1[shuffled]
x2=x2[shuffled]
y=y[shuffled]

train_rate=0.95
dev_num=int(len(x1)*train_rate)
x1_train,x2_train,y_train=x1[:dev_num],x2[:dev_num],y[:dev_num]
x1_dev,x2_dev,y_dev=x1[dev_num:],x2[dev_num:],y[dev_num:]


def get_len(data):
    data_len=[len([v for v in line if v>0])for line in data]
    return data_len


def get_batches(x1,x2,y,batch_size):
    batches_x1=[]
    batches_x2=[]
    batches_y=[]

    shuffled = np.random.permutation(range(len(x1)))
    x1 = x1[shuffled]
    x2 = x2[shuffled]
    y = y[shuffled]

    data_len=len(x1)
    batch_total=int(data_len/batch_size)+1
    start_id=0
    for i in range(batch_total+1):
        batch_x1=x1[start_id:min(start_id+batch_size,data_len)]
        batch_x2=x2[start_id:min(start_id+batch_size,data_len)]
        batch_y=y[start_id:min(start_id+batch_size,data_len)]
        batches_x1.append(batch_x1)
        batches_x2.append(batch_x2)
        batches_y.append(batch_y)
        start_id+=batch_size
    return batches_x1,batches_x2,batches_y


seq_length=30
embedding_size=200
hidden_size=160
n_classes=2
batch_size=200
learning_rate=0.001
optimizer='adam'
l2=0.01
clip_value=5
epoch_num=20

x1_devlen=get_len(x1_dev)
x2_devlen=get_len(x2_dev)

def train():
    sess=tf.Session()
    model=model8.ESIM(seq_length,n_vocab,embedding_size,hidden_size,n_classes,batch_size,learning_rate,optimizer,l2,clip_value)
    sess.run(tf.global_variables_initializer())

    count_num = 0
    for i in range(epoch_num):
        batches_x1,batches_x2,batches_y=get_batches(x1_train,x2_train,y_train,batch_size)

        for x1,x2,y in zip(batches_x1,batches_x2,batches_y):
            x1_len=get_len(x1)
            x2_len=get_len(x2)
            feed_dict={model.premise:x1,
                       model.premise_mask:x1_len,
                       model.hypothesis:x2,
                       model.hypothesis_mask:x2_len,
                       model.y_input:y,
                       model.dropout_keep_prob:0.3}
            _, batch_loss, batch_acc, p= sess.run([model.train, model.loss, model.acc ,model.P], feed_dict=feed_dict)
            print(count_num,' epoch:',i,' loss:',round(batch_loss,3),'acc:',round(batch_acc,3),'P:',p)

            if count_num%20==0:
                dev_feed_dict={model.premise:x1_dev,
                       model.premise_mask:x1_devlen,
                       model.hypothesis:x2_dev,
                       model.hypothesis_mask:x2_devlen,
                       model.y_input:y_dev,
                       model.dropout_keep_prob:1.0}
                batch_loss, batch_acc, p = sess.run([model.loss, model.acc, model.P],feed_dict=dev_feed_dict)
                print('\ndev:')
                print(count_num, ' epoch:', i, ' loss:', round(batch_loss, 3), 'acc:', round(batch_acc, 3), 'P:', p,'\n')

            count_num+=1

train()