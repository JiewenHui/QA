#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
from multiply import ComplexMultiply
import math
from scipy import linalg
# point_wise obbject
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import initializers
# from complexnn.dense import ComplexDense
# from complexnn.utils import GetReal
from keras import backend as K
import math
from QA_CNN_point import QA_quantum

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

class QA_quantum(QA_quantum):
    def __init__(
      self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,embeddings_complex,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,position_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):

        super().__init__(max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,embeddings_complex,dropout_keep_prob,filter_sizes,num_filters,l2_reg_lambda,
      is_Embedding_Needed,trainable,overlap_needed,position_needed ,pooling ,hidden_num,extend_feature_dim)
        
        
    def feed_neural_work(self):
        self.h_drop_out = self.narrow_convolutionandpool_real_imag(tf.expand_dims(self.M_qa_real,-1),tf.expand_dims(self.M_qa_imag,-1))
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[2*self.num_filters_total, 2],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            for p in self.para:
                l2_loss += tf.nn.l2_loss(p)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def narrow_convolutionandpool_real_imag(self,embedding_real,embedding_imag):

        self.num_filters_total = self.embedding_size * self.embedding_size
        self.h_pool_real=tf.reshape(embedding_real, [-1, self.num_filters_total])
        self.h_pool_imag=tf.reshape(embedding_imag, [-1, self.num_filters_total])
        h_drop_real = tf.nn.dropout(self.h_pool_real, self.dropout_keep_prob)
        h_drop_imag = tf.nn.dropout(self.h_pool_imag, self.dropout_keep_prob)
        h_drop=tf.concat([h_drop_real,h_drop_imag],1)
        return h_drop

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.joint_representation()
        self.feed_neural_work()

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                embeddings_complex=None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3*33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y,
            cnn.q_overlap:input_overlap_q,
            cnn.a_overlap:input_overlap_a,
            cnn.q_position:q_posi,
            cnn.a_position:a_posi
        }

        see,question,answer,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.answer,cnn.scores],feed_dict)
        print (see)

# regularizer1 = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            
#             W1 = tf.get_variable( "W_hidden1",
#                 shape=[3*self.num_filters,self.hidden_num],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer1)
#             b1 = tf.get_variable('b_hidden1', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer1)
#             self.para.append(W1)
#             self.para.append(b1)
#             self.hidden_output_pos = tf.nn.tanh(tf.nn.xw_plus_b(self.represent_pos, W1, b1, name = "hidden_output"))

#             W2 = tf.get_variable(
#                 "W_outpu2t",
#                 shape = [self.hidden_num, 2],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer1)
#             b2 = tf.get_variable('b_output2', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer1)
#             self.para.append(W2)
#             self.para.append(b2)

#             self.logits_pos = tf.nn.xw_plus_b(self.hidden_output_pos, W2, b2, name = "scores")
#             self.scores_pos = tf.nn.softmax(self.logits_pos)
#             self.predictions_pos = tf.argmax(self.scores_pos, 1, name = "predictions")

#             regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            
#             W3 = tf.get_variable( "W_hidden3",
#                 #shape=[102,self.hidden_num],
#                 shape=[3*self.num_filters,self.hidden_num],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer)
#             b3 = tf.get_variable('b_hidden3', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer)
#             self.para.append(W3)
#             self.para.append(b3)
            
#             self.hidden_output_neg = tf.nn.tanh(tf.nn.xw_plus_b(self.represent_neg, W3, b3, name = "hidden_output"))
#             W4 = tf.get_variable(
#                 "W_output3",
#                 shape = [self.hidden_num, 2],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer)
#             b4 = tf.get_variable('b_output3', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer)
#             self.para.append(W4)
#             self.para.append(b4)

#             self.logits_neg = tf.nn.xw_plus_b(self.hidden_output_neg, W4, b4, name = "scores")
#             self.scores_neg = tf.nn.softmax(self.logits_neg)
#             self.predictions_neg = tf.argmax(self.scores_neg, 1, name = "predictions")
