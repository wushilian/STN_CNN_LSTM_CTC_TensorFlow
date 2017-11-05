import tensorflow as tf
import numpy as np
import random
import time
import logging,datetime
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import utils
import os,sys
import resnet as res
slim=tf.contrib.slim
from STN import spatial_transformer_network as stn

FLAGS=utils.FLAGS
#26*2 + 10 digit + blank + space
num_classes=utils.num_classes
max_timesteps=0
num_features=utils.num_features

def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    # is_training presents the net training or testing;is_conv_out presents the layer whether convolutioon
    # is_training表示网络是否训练，is_conv_out表示该层是不是卷积层
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 0.001)
def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class, e.g. LSTMCell
    :param num_units: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int
    :return: the output of last layer bidirectional rnn with concatenating
    这里用到几个tf的特性
    1. tf.variable_scope(None, default_name="bidirectional-rnn")使用default_name
    的话,tf会自动处理命名冲突
    """
    # TODO: add time_major parameter, and using batch_size = tf.shape(inputs)[0], and more assert
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        # 恶心的很.如果不在这加的话,会报错的.
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs

class Graph(object):
    def __init__(self,is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, utils.image_width, utils.image_height, 1])
            '''net=res.residual_block(self.inputs,1,64,subsample=True,phase_train=True,scope='res_block1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = res.residual_block(net, 64, 128, subsample=True, phase_train=True, scope='res_block2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = res.residual_block(net, 128, 256, subsample=True, phase_train=True, scope='res_block3')
            net = res.residual_block(net, 256, 256, subsample=False, phase_train=True, scope='res_block4')
            net = res.residual_block(net, 256, 512, subsample=True, phase_train=True, scope='res_block5')
            net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool3')
            net = res.residual_block(net, 512, 512, subsample=False, phase_train=True, scope='res_block6')
            net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
            net = slim.conv2d(net, 512, [2, 2], padding='VALID', scope='conv7')
            net = batch_norm(net, True)'''
            with tf.variable_scope('STN'):
                #Localisation net
                conv1_loc = slim.conv2d(self.inputs, 32, [3, 3], scope='conv1_loc')
                pool1_loc = slim.max_pool2d(conv1_loc, [2, 2], scope='pool1_loc')
                conv2_loc = slim.conv2d(pool1_loc, 64, [3, 3], scope='conv2_loc')
                pool2_loc = slim.max_pool2d(conv2_loc, [2, 2], scope='pool2_loc')
                pool2_loc_flat = slim.flatten(pool2_loc)
                fc1_loc = slim.fully_connected(pool2_loc_flat, 2048, scope='fc1_loc')
                fc2_loc = slim.fully_connected(fc1_loc, 512, scope='fc2_loc')
                fc3_loc = slim.fully_connected(fc2_loc, 6, activation_fn=tf.nn.tanh, scope='fc3_loc')
                # spatial transformer
                h_trans = stn(self.inputs, fc3_loc, (120, 32))
            with tf.variable_scope('CNN'):
                net = slim.conv2d(h_trans, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3')
                net = batch_norm(net, is_training)
                net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='conv5')
                net = batch_norm(net, is_training)
                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
                net = slim.conv2d(net, 512, [2, 2], padding='VALID', scope='conv7')
                net = batch_norm(net, is_training)
            print(net)
            temp_inputs = net
            with tf.variable_scope('BLSTM'):
                self.labels = tf.sparse_placeholder(tf.int32)
                self.lstm_inputs = tf.reshape(temp_inputs, [-1, 27, 512])
                # 1d array of size [batch_size]
                self.seq_len = tf.placeholder(tf.int32, [None])
                # Defining the cell
                # Can be:
                #   tf.nn.rnn_cell.RNNCell
                #   tf.nn.rnn_cell.GRUCell
                # cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
                # cell = tf.contrib.rnn.DropoutWrapper(cell = cell,output_keep_prob=0.8)
                #
                # cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
                # cell1 = tf.contrib.rnn.DropoutWrapper(cell = cell1,output_keep_prob=0.8)
                # Stacking rnn cells
                # stack = tf.contrib.rnn.MultiRNNCell([cell,cell1] , state_is_tuple=True)

                # stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(FLAGS.num_hidden,state_is_tuple=True) for _ in range(FLAGS.num_layers)] , state_is_tuple=True)
                outputs = stacked_bidirectional_rnn(tf.contrib.rnn.LSTMCell, FLAGS.num_hidden, 2, self.lstm_inputs,self.seq_len)
            # The second output is the last state and we will no use that
            # outputs, _ = tf.nn.dynamic_rnn(stack, self.lstm_inputs, self.seq_len, dtype=tf.float32)
            shape = tf.shape(self.lstm_inputs)
            batch_s, max_timesteps = shape[0], shape[1]
            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,num_classes],stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))
            logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,self.global_step,FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True)
           # self.optimizer=tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
            # momentum=FLAGS.momentum).minimize(self.cost,global_step=self.global_step)

            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum, use_nesterov=True).minimize(self.cost,
                                                                                                            global_step=self.global_step)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
            # beta1=FLAGS.beta1,beta2=FLAGS.beta2).minimize(self.loss,global_step=self.global_step)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            # Inaccuracy: label error rate
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            # tf.summary.scalar('lerr',self.lerr)
            self.merged_summay = tf.summary.merge_all()
