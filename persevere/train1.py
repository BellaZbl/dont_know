import tensorflow as tf
from persevere import model6
from tensorflow.contrib import learn
import numpy as np
import pickle

vocab_processor=learn.preprocessing.VocabularyProcessor.restore('vocab_processor.model')
vocab_num=len(vocab_processor.vocabulary_)

x1=pickle.load(open('./data/seq1.pkl','rb'))
x2=pickle.load(open('./data/seq2.pkl','rb'))
y=pickle.load(open('./data/lables.pkl','rb'))

train_rate=0.9
dev_num=int(len(x1)*train_rate)
x1Train=x1[:dev_num]
x2Train=x2[:dev_num]
yTrain=y[:dev_num]

x1Dev=x1[dev_num:]
x2Dev=x2[dev_num:]
yDev=y[dev_num:]

dropout_keep_prob=0.3
batch_size=5000
num_epochs=10
dev_batch_size=1000
evaluate_every=200

def train():
    with tf.Graph().as_default():
        sess=tf.Session()
        siamese_cnn=model6.siameseTextCNN(30,30,None,vocab_num,200,60,[2,3,5])

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)

        grads_and_vars = optimizer.compute_gradients(siamese_cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        def train_step(x1Batch, x2Batch, yBatch):
            feed_dict = {
                siamese_cnn.s1: x1Batch,
                siamese_cnn.s2: x2Batch,
                siamese_cnn.input_y: yBatch,
                siamese_cnn.dropout_keep_prob:dropout_keep_prob
            }
            _, step, loss, accuracy,pre,rec,positive,losses = sess.run([train_op, global_step, siamese_cnn.loss, siamese_cnn.accuracy,siamese_cnn.tf_precision,siamese_cnn.tf_recall,siamese_cnn.P,siamese_cnn.losses],feed_dict)
            print(losses)
            print("step {}, loss {:g}, acc {:g},     precision：{:g}, recall：{:g},positive:{:g}".format(step, loss, accuracy,pre,rec,positive))

        batches = batch_iter(list(zip(x1Train, x2Train, yTrain)), batch_size, num_epochs)

        def dev_step(x1Batch, x2Batch, yBatch):
            feed_dict = {
                siamese_cnn.s1: x1Batch,
                siamese_cnn.s2: x2Batch,
                siamese_cnn.input_y: yBatch,
                siamese_cnn.dropout_keep_prob:1.0
            }
            _, step, loss, accuracy= sess.run([train_op, global_step, siamese_cnn.loss, siamese_cnn.accuracy],feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        def dev_test():
            batches_dev = batch_iter(list(zip(x1Dev, x2Dev, yDev)), dev_batch_size, 1)
            for batch_dev in batches_dev:
                x1BatchDev, x2BatchDev, yBatchDev = zip(*batch_dev)
                dev_step(x1BatchDev, x2BatchDev, yBatchDev)

        for batch in batches:
            x1BatchTrain, x2BatchTrain, yBatchTrain = zip(*batch)
            train_step(x1BatchTrain, x2BatchTrain, yBatchTrain)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_test()


def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    train()