#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np

# point_wise obbject
class QA(object):
    def __init__(
      self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,position_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):

        
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.overlap_needed = overlap_needed
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.filter_sizes = filter_sizes
        self.pooling = pooling
        self.position_needed = position_needed
        if self.overlap_needed:
            self.total_embedding_dim = embedding_size + extend_feature_dim
        else:
            self.total_embedding_dim = embedding_size
        if self.position_needed:
            self.total_embedding_dim = self.total_embedding_dim + extend_feature_dim
        print self.total_embedding_dim
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.para = []
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.hidden_num = 50
        self.extend_feature_dim = extend_feature_dim
        self.is_Embedding_Needed = is_Embedding_Needed   
        

        #the parameter for density matrix
        self.density_filter_sizes = [20]
        self.density_num_filters = 10
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.q_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_feature_embeding')
        self.a_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_feature_embeding')
        self.q_position = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position')
        self.a_position = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_position')
    def add_embeddings(self):

        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.overlap_W = tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            self.position_W = tf.Variable(tf.random_uniform([300,self.extend_feature_dim], -1.0, 1.0),name = 'W',trainable = True)
            self.para.append(self.embedding_W)
            self.para.append(self.overlap_W)

        #get embedding from the word indices
        self.embedded_chars_q = self.concat_embedding(self.question,self.q_overlap,self.q_position)
        self.embedded_chars_a = self.concat_embedding(self.answer,self.a_overlap,self.a_position)
    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]),name = 'weighted_q')
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]),name = 'weighted_a')
    def convolution(self):
        #initialize my conv kernel
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,self.num_filters]
                W = tf.get_variable('W' + str(i),filter_shape,tf.float32,tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_IN',uniform=True))
                b = tf.get_variable('b' + str(i),[self.num_filters],tf.float32,tf.constant_initializer(0.01) )
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                # b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name = "b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.q_conv = self.wide_convolution(self.embedded_chars_q)
        self.a_conv = self.wide_convolution(self.embedded_chars_a)


    def pooling_graph(self):
        with tf.name_scope('pooling'):
            #different pooling strategy. max pooing or attentive pooling
            if self.pooling == 'max':
                self.q_pooling = tf.reshape(self.max_pooling(self.q_conv,self.max_input_left),[-1, self.num_filters_total])
                self.a_pooling = tf.reshape(self.max_pooling(self.a_conv,self.max_input_right),[-1,self.num_filters_total])
            elif self.pooling == 'attentive':
                with tf.name_scope('attention'):    
                    self.U = tf.Variable(tf.truncated_normal(shape = [self.num_filters_total,self.num_filters_total],stddev = 0.01,name = 'U'))
                    self.para.append(self.U)

                self.q_pooling,self.a_pooling = self.attentive_pooling(self.q_conv,self.a_conv)
        

                self.q_pooling = tf.reshape(self.q_pooling,[-1,self.num_filters_total])
                self.a_pooling = tf.reshape(self.a_pooling,[-1,self.num_filters_total])
            else:
                print 'no pooling'
    def out_product_interact(self):

   

        # self.density_q = self.density_matrix(self.q_pooling)
        # self.density_a = self.density_matrix(self.a_pooling)
        print self.q_conv
        print self.a_conv
        exit()
        self.density_q = self.density_matrix(self.q_conv,self.weighted_q)
        self.density_a = self.density_matrix(self.a_conv,self.weighted_a)
        self.M_qa = tf.matmul(self.density_q,self.density_a)
    def density_matrix(self,sentence,weighted_sentence):
        # sentence = tf.expand_dims(sentence,-1)
        # sentence = tf.nn.l2_normalize(sentence,1)
        # reverse = tf.transpose(sentence,[0,2,1])
        sentence = tf.nn.l2_normalize(sentence,3)
        reverse = tf.transpose(sentence,[0,1,3,2])
        mul = tf.matmul(reverse,sentence)
        pi_mul = tf.multiply(mul,tf.nn.softmax(weighted_sentence,1))
        self.see = self.weighted_q
        # self.see = tf.reduce_sum(reverse,2)
        return tf.reduce_sum(pi_mul,1)
    
        # return tf.matmul(sentence,reverse)
    def trace_represent(self):
        self.density_diag = tf.matrix_diag_part(self.M_qa)
        self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        self.match_represent = tf.concat([self.density_diag],1)

    def density_convolution(self):
        print 'density convolution'
        #initialize my conv kernel
        self.density_kernels = []
        for i,filter_size in enumerate(self.density_filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,filter_size,1,self.density_num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                b = tf.Variable(tf.constant(0.0, shape = [self.density_num_filters]), name = "b")
                self.density_kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        self.density_num_filters_total = self.density_num_filters * len(self.density_filter_sizes)
        self.qa = self.density_wide_convolution(tf.expand_dims(self.M_qa,-1))
        self.q = self.density_wide_convolution(tf.expand_dims(self.density_q,-1))
        self.a = self.density_wide_convolution(tf.expand_dims(self.density_a,-1))
        print self.M_qa
        print self.qa
    def density_max_pooling(self,conv):

        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, self.num_filters_total, self.num_filters_total, 1],
                    # ksize = [1, self.max_input_left, self.max_input_right, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def density_pooling_graph(self):
        print 'density pooling'
        with tf.name_scope('pooling'):
            matrix_list = []
            for matrix in [self.qa]:
                matrix_list.append(tf.reshape(self.density_max_pooling(matrix),[-1,self.density_num_filters_total]))
            # self.pooling = self.density_max_pooling(self.qa)
            self.represent = tf.concat(matrix_list,1)

            # self.represent = tf.reshape(self.pooling,[-1,self.density_num_filters_total])
            # raw_pooling = tf.contrib.layers.flatten(tf.reduce_sum(self.qa,1))
            # col_pooling = tf.contrib.layers.flatten(tf.reduce_sum(self.qa,2))
            # self.represent = tf.concat([raw_pooling,col_pooling],1)
            print self.represent

    def density_wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.density_filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.density_kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )

            h = tf.nn.relu(tf.nn.bias_add(conv, self.density_kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)

        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped


    def interact(self):
        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_filters_total],
                initializer = tf.contrib.layers.xavier_initializer())
            # print 'q_pooling',self.q_pooling
            # print 'num_filters',self.num_filters_total
            self.transform_left = tf.matmul(self.q_pooling, W)
            print 'transform_left',self.transform_left
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.a_pooling), 1, keep_dims=True)
            self.para.append(W)
            print W
        # concat the input vector to classification task
        self.feature = tf.concat([self.q_pooling,self.a_pooling],1,name = 'feature')

        # self.feature = tf.concat([self.q_pooling,self.sims,self.a_pooling],1,name = 'feature')
    def feed_neural_work(self):
        print 'feed_neural work'
        with tf.name_scope('neural_network'):
            W = tf.get_variable(
                "W_hidden",
                shape = [self.num_filters_total,self.hidden_num],
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.constant_initializer(0.01))
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.match_represent, W, b, name = "hidden_output"))
        #add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape = [self.hidden_num, 2],
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer())
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
            
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def concat_embedding(self,words_indice,overlap_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        if not self.overlap_needed :
            if not self.position_needed:
                return tf.expand_dims(embedded_chars_q,-1)
            else:
                return tf.expand_dims(tf.concat([embedded_chars_q,position_embedding],2),-1)
        else:
            if not self.position_needed:
                return  tf.expand_dims(tf.concat([embedded_chars_q,overlap_embedding_q],2),-1)
            else:
                return tf.expand_dims(tf.concat([embedded_chars_q,overlap_embedding_q,position_embedding],2),-1)
        

    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def attentive_pooling(self,input_left,input_right):
        Q = tf.reshape(input_left,[self.batch_size,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        A = tf.reshape(input_right,[self.batch_size,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')

        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        first = tf.matmul(tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]),self.U)
        second_step = tf.reshape(first,[self.batch_size,-1,len(self.filter_sizes) * self.num_filters])
        result = tf.matmul(second_step,tf.transpose(A,perm = [0,2,1]))
        G = tf.tanh(result)
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

        attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')

        R_q = tf.reshape(tf.matmul(Q,attention_q,transpose_a = 1),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_q')
        R_a = tf.reshape(tf.matmul(attention_a,A),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_a')

        return R_q,R_a
    
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_dim, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name = "relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        # cnn_reshaped = tf.concat(3,cnn_outputs)
        return cnn_outputs
        # return cnn_reshaped
    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.convolution()
        self.pooling_graph()
        self.out_product_interact()
        self.trace_represent()

        # self.density_convolution()
        # self.density_pooling_graph()
        # self.interact()
        self.feed_neural_work()
        self.create_loss()

if __name__ == '__main__':
    max_sentence = 50
    cnn = QA(max_input_left = max_sentence,
                max_input_right = max_sentence,
                vocab_size = 5000,
                embedding_size = 100,
                batch_size = 3,
                embeddings = None,
                dropout_keep_prob = 1,
                filter_sizes = [3,4,5],
                num_filters = 64,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = True)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * max_sentence),[3,max_sentence])
    input_x_2 = np.reshape(np.arange(3 * max_sentence),[3,max_sentence])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,max_sentence))
    input_overlap_a = np.ones((3,max_sentence))
    q_posi = np.ones((3,max_sentence))
    a_posi = np.ones((3,max_sentence))
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
       
        question,answer,scores,density_q = sess.run([cnn.question,cnn.answer,cnn.scores,cnn.density_q],feed_dict)
        print scores

       
