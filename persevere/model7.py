import tensorflow as tf


class SiameseLSTM(object):

    def BiRNN(self, x, dropout, scope, hidden_units):
        n_hidden = hidden_units
        n_layers = 3

        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = 3.0 * y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_units, batch_size):

        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),trainable=True, name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", hidden_units)
            self.out2 = self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", hidden_units)

            #  这里计算的是欧式距离 前两个distance的shape都是[batch_size,1]，最后一个distance是[batch_size,]
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("loss"):
            #  这里用的是对比损失
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)

        with tf.name_scope("accuracy"):
            self.predictions=tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance))
            TP = tf.count_nonzero(self.predictions*self.input_y)
            FP = tf.count_nonzero(self.predictions * (self.input_y - 1))
            FN = tf.count_nonzero((self.predictions - 1) * self.input_y)

            self.P=(TP+FP)
            self.tf_precision = TP / (TP + FP)
            self.tf_recall = TP / (TP + FN)
            self.accuracy = 2 * self.tf_precision * self.tf_recall / (self.tf_precision + self.tf_recall)