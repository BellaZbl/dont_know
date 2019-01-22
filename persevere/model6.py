import tensorflow as tf
import numpy as np
from sklearn import metrics

class siameseTextCNN(object):

    def __init__(self,seq1_len,seq2_len,w2v_model,vocabSize, embeddingSize,numFilters,filterSizes,numHidden=10,l2_reg_lambda=0.05):

        self.s1 = tf.placeholder(tf.int32, [None, seq1_len], name="input_s1")
        self.s2 = tf.placeholder(tf.int32, [None, seq2_len], name="input_s2")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")
        self.y=tf.one_hot(self.input_y,2)
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='drop_out_keep_prob')

        l2_reg = tf.constant(0.0)
        maxLenX1 = seq1_len
        maxLenX2 = seq2_len
        if w2v_model is None:
            self.W = tf.Variable(tf.random_uniform([vocabSize, embeddingSize], -1.0, 1.0),name="word_embeddings")
        else:
            self.W = tf.get_variable("word_embeddings", initializer=w2v_model.vectors.astype(np.float32))

        self.embeddedChars1 = tf.expand_dims(tf.nn.embedding_lookup(self.W, self.s1), -1)
        self.embeddedChars2 = tf.expand_dims(tf.nn.embedding_lookup(self.W, self.s2), -1)

        output1 = []
        output2 = []
        numFiltersTotal = numFilters * len(filterSizes)
        for i, filterSize in enumerate(filterSizes):
            filterShape = [filterSize, embeddingSize, 1, numFilters]
            for k in [1, 2]:
                with tf.name_scope("Conv-Maxpool-Layer-%s-%s" % (str(k), filterSize)):
                    W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(1.0, shape=[numFilters]), name="b")
                    conv = tf.nn.conv2d(eval('self.embeddedChars' + str(k)), W,strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, eval('maxLenX' + str(k)) - filterSize + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    eval('output' + str(k) + '.append(pooled)')

        self.hiddenPooled1 = tf.reshape(tf.concat(output1, -1), [-1, numFiltersTotal], name='hiddenPooled1')
        self.hiddenPooled2 = tf.reshape(tf.concat(output2, -1), [-1, numFiltersTotal], name='hiddenPooled2')

        self.hiddenPooled1=tf.nn.dropout(self.hiddenPooled1,self.dropout_keep_prob)
        self.hiddenPooled2 = tf.nn.dropout(self.hiddenPooled2, self.dropout_keep_prob)

        with tf.name_scope("similarity"):
            W = tf.get_variable("W",shape=[numFiltersTotal, numFiltersTotal],initializer=tf.contrib.layers.xavier_initializer())
            self.transform1 = tf.matmul(self.hiddenPooled1, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform1, self.hiddenPooled2), 1, keep_dims=True)

        l2_loss = tf.constant(0.0)
        self.Input = tf.concat([self.hiddenPooled1, self.sims, self.hiddenPooled2], 1, name='Input')

        with tf.name_scope("hidden"):
            W = tf.get_variable("W_hidden",shape=[2 * numFiltersTotal + 1, numHidden],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[numHidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hiddenOutput = tf.nn.relu(tf.nn.xw_plus_b(self.Input, W, b, name="hiddenOutput"))

        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hiddenOutput, self.dropout_keep_prob, name="hidden_output_drop")

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[numHidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.hDrop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            self.soft=tf.nn.softmax(logits=self.scores)
            coe = tf.constant([1.0, 2.0])
            y_coe = tf.cast(self.y * coe,tf.float32)
            losses = -tf.reduce_mean(y_coe * tf.log(self.soft+0.0000001))
            self.losses=losses
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            actual=tf.cast(self.input_y,tf.int64)
            TP = tf.count_nonzero(self.predictions*actual)
            FP = tf.count_nonzero(self.predictions * (actual - 1))
            FN = tf.count_nonzero((self.predictions - 1) * actual)

            self.P=(TP+FP)
            self.tf_precision = TP / (TP + FP)
            self.tf_recall = TP / (TP + FN)
            self.accuracy = 2 * self.tf_precision * self.tf_recall / (self.tf_precision + self.tf_recall)





