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

class CNN(QA_quantum):
    def __init__(
      self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,embeddings_complex,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,position_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):

        super().__init__(max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,embeddings_complex,dropout_keep_prob,filter_sizes,num_filters,l2_reg_lambda,
      is_Embedding_Needed,trainable,overlap_needed,position_needed ,pooling ,hidden_num,extend_feature_dim)
 
        
    def feed_neural_work(self):
        with tf.name_scope('regression'):
       
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            W = tf.get_variable( "W_hidden",
                #shape=[102,self.hidden_num],
                shape=[2*self.num_filters_total+2,self.hidden_num],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.tanh(tf.nn.xw_plus_b(self.represent, W, b, name = "hidden_output"))
            #self.hidden_output=tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            W = tf.get_variable(
                "W_output",
                shape = [self.hidden_num, 2],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.hidden_output, W, b, name = "scores")
            print(self.logits)
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
    
    def convolution(self):
        # print('self.M_qa_real',self.M_qa_real)
        # print('self.M_qa_imag',self.M_qa_imag)
        # print('self.overlap',self.overlap)
        # exit()
        self.h_pool_real = []
        self.h_pool_imag = []
        for i in range(self.batch_size):
            temp = tf.diag_part(self.M_qa_real[i])
            temp = tf.reshape(temp,[1,-1])
            self.h_pool_real.append(temp)
        self.h_pool_real = tf.reshape(self.h_pool_real,[self.batch_size,-1])    
        for i in range(self.batch_size):
            temp = tf.diag_part(self.M_qa_imag[i])
            temp = tf.reshape(temp,[1,-1])
            self.h_pool_imag.append(temp)
        self.h_pool_imag = tf.reshape(self.h_pool_imag,[self.batch_size,-1]) 
        self.represent = tf.concat([self.h_pool_real,self.h_pool_imag,self.overlap],1)

        # print('self.h_pool_real',self.h_pool_real)
        # print('self.h_pool_imag',self.h_pool_imag)
        # print('self.overlap',self.overlap)
        # print('self.represent',self.represent)
        # exit()
        self.num_filters_total = self.embedding_size
        # self.qa_real = self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)-self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)
        # print(self.qa_real)
        # self.qa_imag = self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)+self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)
        # print(self.qa_imag)


    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.joint_representation()
        # self.trace_represent()
        self.convolution()
        self.feed_neural_work()
        self.create_loss()

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
