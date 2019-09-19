#import progressbar
import random
import re
import sys
import time
import os
import os.path
import string
import subprocess

import numpy as np
from scipy import stats, spatial
from sklearn.decomposition import TruncatedSVD
#import matplotlib.pyplot as plt

import tensorflow as tf






'''
Here we use CPU instead of GPU because according to experiments,
it takes much less time to run on CPU, possibly because of the
high cost of GPU-CPU traffic
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'







# load data

def load_sent(path, max_size=4):
    data = []
    with open(path) as f:
        for line in f:
            tmp = line.split()
            data.append(tmp)
    return data


#root_ner = "./data/ner-data/"
root_ner = "../../data/ner/ner_data_richard/"

dev_data = load_sent(root_ner+"eng.dev.bioes.conll")
train_data = load_sent(root_ner+"eng.train.bioes.conll")
test_data = load_sent(root_ner+"eng.test.bioes.conll")







def load_embedding(emb_file):
    data = []
    embedding2id = {'<pad>':0, '<s>':1, '</s>':2}
    # word2id = {}
    id2embedding = ['<pad>', '<s>', '</s>']
    embedding_old = {}
    # id2word = []
    with open(emb_file) as f:
        for line in f:
            try:
                parts = line.split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]])
                embedding2id[word] = len(embedding2id)
                id2embedding.append(word)
                embedding_old[word] = vec
            except:
                print(len(embedding2id))
    return embedding2id,id2embedding, embedding_old

#embedding2id, id2embedding, embedding_old = load_embedding("/share/data/speech/Data/zeweichu/GloVe/glove.6B.100d.txt") # change embedding_global
embedding2id, id2embedding, embedding_old = load_embedding("../../GloVe/glove.6B.100d.txt") 
#embedding2id, id2embedding, embedding_old = load_embedding("/share/data/speech/Data/zeweichu/GloVe/glove.6B.100d.txt") # change embedding_global









def convert_tag_to_id(X, which):
    tag2id = {'<pad>':0, '<s>':1, '</s>':2}
    id2tag = ['<pad>', '<s>', '</s>']
    for x in X:
        if len(x) != 0:
            try:
                tmp = x[which]
            except:
                print(x)
            if tmp not in tag2id:
                tag2id[tmp] = len(tag2id)
                id2tag.append(tmp)
    return (tag2id, id2tag)

tag2id, id2tag = convert_tag_to_id(train_data+dev_data+test_data, 3)
pos2id, id2pos = convert_tag_to_id(train_data+dev_data+test_data, 1)
chunk2id, id2chunk = convert_tag_to_id(train_data+dev_data+test_data, 2)











def preprocess_data_according_to_rules(data):
    for i in range(len(data)):
        if len(data[i]) > 0:
            if not data[i][0].islower():
                data[i].append(1) # there is at least one char upper case
            else:
                data[i].append(0) # all lower case
            
            def is_number(s):
                tmp_tf = 0
                for c in s:
                    if '0' <= c <= '9':
                        tmp_tf = 1
                if tmp_tf:
                    for c in s:
                        if not ('0' <= c <= '9' or c == '-' or c == ',' or c == '.'):
                            return False
                    return True
                else:
                    return False
            
            if is_number(data[i][0]):
                data[i].append(1)
            else:
                data[i].append(0)
            
            tmp = data[i][0].lower()
            
#             def is_number(s):
#                 try:
#                     float(s)
#                     return True
#                 except ValueError:
#                     return False

            if tmp not in embedding_old:
                data[i][0] = 'unk'
            else:
                data[i][0] = tmp
    return data



train_data = preprocess_data_according_to_rules(train_data)
dev_data = preprocess_data_according_to_rules(dev_data)
test_data = preprocess_data_according_to_rules(test_data)













def construct_word_id():
    data = train_data+dev_data+test_data
    word2id = {'<pad>':0,'<s>':1,'</s>':2} # UUUNKKK replaced by unk
    id2word = ['<pad>','<s>','</s>']
    for i in range(len(data)):
        if len(data[i]) > 0:
            tmp = data[i][0]
            if tmp not in word2id:
                word2id[tmp] = len(word2id)
                id2word.append(tmp)
    return word2id, id2word


word2id, id2word = construct_word_id() # 6603 vocab size
len(word2id)













def construct_embedding(embedding_size, dim_emb):
    #embedding = np.random.random_sample((embedding_size, dim_emb)) - 0.5
    embedding = np.zeros((embedding_size, dim_emb))
    for word in word2id:
        try: 
            embedding[word2id[word]] = embedding_old[word]
        except:
            print(word)
    return embedding

embedding_global = construct_embedding(len(word2id), 100) # 300 from GloVe











# data preprocessing and batch generation 

# get a list of x and y
# turn unknown word to UUUNKKK
def turn_data_into_x_y(dataset, word2id):
    x, y, pos, chunk, case, num = [], [], [], [], [], []
    x_tmp, y_tmp, pos_tmp, chunk_tmp, case_tmp, num_tmp = ['<s>'], [], [], [], [], []
    for i in range(len(dataset)):
        tup = dataset[i]
        if len(tup) == 6:
            word = tup[0] #.lower()
            if word not in word2id:
                print(word)
            x_tmp.append(word)
            y_tmp.append(tup[3])
            pos_tmp.append(tup[1])
            chunk_tmp.append(tup[2])
            case_tmp.append(tup[4])
            num_tmp.append(tup[5])
        elif len(tup) == 0:
            x_tmp.append('</s>')
            x.append(x_tmp)
            y.append(y_tmp)
            pos.append(pos_tmp)
            chunk.append(chunk_tmp)
            case.append(case_tmp)
            num.append(num_tmp)
            x_tmp, y_tmp, pos_tmp, chunk_tmp, case_tmp, num_tmp = ['<s>'], [], [], [], [], []
        else:
            print("error at index", i)
    return x, y, pos, chunk, case, num


x_train, y_train, pos_train, chunk_train, case_train, num_train = turn_data_into_x_y(train_data, word2id)
x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev = turn_data_into_x_y(dev_data, word2id)
x_test, y_test, pos_test, chunk_test, case_test, num_test = turn_data_into_x_y(test_data, word2id)







all_chars = ['<padunk>']+list(string.punctuation+string.ascii_lowercase+string.digits)
id2char = all_chars
char2id = {}
for x in all_chars:
    char2id[x] = all_chars.index(x)









def construct_embedding_char():
    embedding_char = np.zeros((len(id2char), 16))
    return embedding_char
    
embedding_char_global = construct_embedding_char()







# BiLSTM


def create_cell(dim, dropout):
    cell = tf.nn.rnn_cell.LSTMCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def create_model(sess, dim_h, n_tag, load_model=False, model_path=''):
    model = Model(dim_h, n_tag)
    if load_model:
        print('Loading model from ...')
        model.saver.restore(sess, model_path)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model

def feed_dictionary(model, batch, dropout, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights']}
    return feed_dict

















# SPEN


# spen infnet + tlm


def get_batch(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id):
    pad = word2id['<pad>']
    pad_tag = tag2id['<pad>']
    inputs_x, outputs_y, tlm_outputs_y, weights, tlm_weights, tlm_targets_pos = [], [], [], [], [], []
    inputs_x_char = []
    next_inputs_x = []
    inputs_pos, inputs_chunk, inputs_case, inputs_num = [], [], [], []
    
    inputs_x_reverse, outputs_y_reverse, tlm_outputs_y_reverse, tlm_targets_pos_reverse = [], [], [], []
    next_inputs_x_reverse = []
    inputs_pos_reverse, inputs_chunk_reverse, inputs_case_reverse, inputs_num_reverse = [], [], [], []
    
    len_char = []
    
    max_len = max([len(sent) for sent in x])
    for i in range(len(x)):
        line = x[i] # sentence with start and end symbols
        l = len(line) # sentence length
        padding = [pad] * (max_len - l)
        padding_plus_one = [pad] * (max_len - l + 1)
        padding_tag = [pad_tag] * (max_len - l)
        padding_tag_plus_one = [pad_tag] * (max_len - l + 1)
        tmp_line_x, tmp_line_y, tmp_line_pos, tmp_line_chunk, tmp_line_case, tmp_line_num = [], [], [], [], [], []
        tmp_cline_x = []
        tmp_tmp_cline_x = []
        for word in line:
            tmp_line_x.append(word2id[word])
            for c in word:
                try:
                    tmp_tmp_cline_x.append(char2id[c])
                except:
                    print('char does not exist', c)
            tmp_cline_x.append(tmp_tmp_cline_x)
            tmp_tmp_cline_x = []
        for tag in y[i]:
            tmp_line_y.append(tag2id[tag])
        for pos_single in pos[i]:
            tmp_line_pos.append(pos2id[pos_single])
        for chunk_single in chunk[i]:
            tmp_line_chunk.append(chunk2id[chunk_single])
        for case_single in case[i]:
            tmp_line_case.append(case_single)
        for num_single in num[i]:
            tmp_line_num.append(num_single)
        
        inputs_x.append(tmp_line_x+padding)
        inputs_x_char.append(tmp_cline_x)
        len_char.append(len(tmp_cline_x))
        inputs_x_reverse.append(tmp_line_x[::-1]+padding)
        
        next_inputs_x.append(tmp_line_x[1:]+padding_plus_one)
        next_inputs_x_reverse.append(tmp_line_x[::-1][1:]+padding_plus_one)
        
        outputs_y.append([tag2id['<s>']]+tmp_line_y+[tag2id['</s>']]+padding_tag)
        outputs_y_reverse.append([tag2id['</s>']]+tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag)
        
        inputs_pos.append([pos2id['<s>']]+tmp_line_pos+[pos2id['</s>']]+padding_tag)
        inputs_pos_reverse.append([pos2id['</s>']]+tmp_line_pos[::-1]+[pos2id['<s>']]+padding_tag)
        inputs_chunk.append([chunk2id['<s>']]+tmp_line_chunk+[chunk2id['</s>']]+padding_tag)
        inputs_chunk_reverse.append([chunk2id['</s>']]+tmp_line_chunk[::-1]+[chunk2id['<s>']]+padding_tag)
        
        inputs_case.append([0]+tmp_line_case+[0]+padding_tag)
        inputs_case_reverse.append([0]+tmp_line_case[::-1]+[0]+padding_tag)
        inputs_num.append([0]+tmp_line_num+[0]+padding_tag)
        inputs_num_reverse.append([0]+tmp_line_num[::-1]+[0]+padding_tag)
        
        tlm_outputs_y.append(tmp_line_y+[tag2id['</s>']]+padding_tag_plus_one)
        tlm_outputs_y_reverse.append(tmp_line_y[::-1]+[tag2id['<s>']]+padding_tag_plus_one)
        tlm_targets_pos.append(tmp_line_pos+[tag2id['</s>']]+padding_tag_plus_one)
        tlm_targets_pos_reverse.append(tmp_line_pos[::-1]+[tag2id['<s>']]+padding_tag_plus_one)
        
        weights.append([1.0] * (l+1-1) + [0.0] * (max_len-l))
        tlm_weights.append([1.0] * (l-1) + [0.0] * (max_len-l+1))
        
    tmp_random = random.randint(0,1)
    if tmp_random:
        s1 = np.random.dirichlet(np.ones(len(tag2id))*10,size=1)-float(1)/len(tag2id) # size here represent batch size!!!
        s2 = -(np.random.dirichlet(np.ones(len(tag2id))*10,size=1)-float(1)/len(tag2id))
        s = (s1+s2)/2
    else:
        s = np.zeros((1,len(tag2id)))
        # weights_reverse and tlm_weights_reverse are the same as weights and tlm_weights
        
 
    # wrong: should run CNN through chars of each word

    for i in range(len(inputs_x_char)):
        for j in range(len(inputs_x_char[i])):
            max_len_char = max([len(x) for x in inputs_x_char[i]])
            new_len_char = [(max_len_char - len(inputs_x_char[i][j])) for j in range(len(inputs_x_char[i]))]
            inputs_x_char[i][j] += [0 for k in range(new_len_char[j])]
        
    return {'enc_inputs': inputs_x,
            'enc_inputs_reverse': inputs_x_reverse,
            'enc_inputs_char': inputs_x_char, # batchsize=1
            'next_enc_inputs': next_inputs_x,
            'next_enc_inputs_reverse': next_inputs_x_reverse,
            'inputs_pos': inputs_pos,
            'inputs_pos_reverse': inputs_pos_reverse,
            'inputs_chunk': inputs_chunk,
            'inputs_chunk_reverse': inputs_chunk_reverse,
            'inputs_case': inputs_case,
            'inputs_case_reverse': inputs_case_reverse,
            'inputs_num': inputs_num,
            'inputs_num_reverse': inputs_num_reverse,
            'targets': outputs_y,
            'targets_reverse': outputs_y_reverse,
            'tlm_targets': tlm_outputs_y,
            'tlm_targets_reverse': tlm_outputs_y_reverse,
            'tlm_targets_pos': tlm_targets_pos,
            'tlm_targets_pos_reverse': tlm_targets_pos_reverse,
            'batch_size': len(inputs_x),
            'weights': weights,
            'tlm_weights': tlm_weights,
            'len': max_len,
            'size': len(x),
            'perturb': s}

def get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    n = len(x)
    order = range(n)
    z = sorted(zip(order, x, y, pos, chunk, case, num), key=lambda i: len(i[1]))
    order, x, y, pos, chunk, case, num = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        if (s-t) < batch_size:
            s = t-batch_size
        batches.append(get_batch(x[s:t], y[s:t], pos[s:t], chunk[s:t], case[s:t], num[s:t], word2id, tag2id, pos2id, chunk2id))
        s = t

    return batches, order

















def feed_dictionary(model, batch, dropout, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.enc_inputs_char: batch['enc_inputs_char'],
                 model.enc_inputs_reverse: batch['enc_inputs_reverse'],
                 model.next_enc_inputs: batch['next_enc_inputs'],
                 model.next_enc_inputs_reverse: batch['next_enc_inputs_reverse'],
                 model.inputs_pos: batch['inputs_pos'],
                 model.inputs_pos_reverse: batch['inputs_pos_reverse'],
                 model.inputs_chunk: batch['inputs_chunk'],
                 model.inputs_chunk_reverse: batch['inputs_chunk_reverse'],
                 model.inputs_case: batch['inputs_case'],
                 model.inputs_case_reverse: batch['inputs_case_reverse'],
                 model.inputs_num: batch['inputs_num'],
                 model.inputs_num_reverse: batch['inputs_num_reverse'],
                 model.targets: batch['targets'],
                 model.targets_reverse: batch['targets_reverse'],
                 model.tlm_targets: batch['tlm_targets'],
                 model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                 model.tlm_targets_pos: batch['tlm_targets_pos'],
                 model.tlm_targets_pos_reverse: batch['tlm_targets_pos_reverse'],
                 model.weights: batch['weights'],
                 model.tlm_weights: batch['tlm_weights'],
                 model.perturb: batch['perturb']}
    return feed_dict



def create_cell_gru(dim, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    return cell







def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def cnn(inp, scope, reuse=False):
    filter_sizes = [1,2,3]
    n_filters = 64
    dropout = True
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)
    num_words = inp.get_shape().as_list()[0]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 32])
            b = tf.get_variable('b', [32])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1, 32])

    return logits







# InfNet

class InfNet_TLM(object):
    
    # dim_h 100, tag_size 28
    def __init__(self, dim_h, tag_size, pos_size, chunk_size, vocab_size):

        dim_emb = 100
        beta1, beta2 = 0.9, 0.999
        dim_d = 2*dim_h # value of d
        
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_len = tf.placeholder(tf.int32, name='batch_len')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None], name='enc_inputs') # size * len
        self.enc_inputs_reverse = tf.placeholder(tf.int32, [None, None], name='enc_inputs_reverse')
        self.next_enc_inputs = tf.placeholder(tf.int32, [None, None], name='next_enc_inputs') # size * len
        self.next_enc_inputs_reverse = tf.placeholder(tf.int32, [None, None], name='next_enc_inputs_reverse')
        self.inputs_pos = tf.placeholder(tf.int32, [None, None], name='inputs_pos')
        self.inputs_pos_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_pos_reverse')
        self.inputs_chunk = tf.placeholder(tf.int32, [None, None], name='inputs_chunk')
        self.inputs_chunk_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_chunk_reverse')
        self.inputs_case = tf.placeholder(tf.int32, [None, None], name='inputs_case')
        self.inputs_case_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_case_reverse')
        self.inputs_num = tf.placeholder(tf.int32, [None, None], name='inputs_num')
        self.inputs_num_reverse = tf.placeholder(tf.int32, [None, None], name='inputs_num_reverse')
        
        self.enc_inputs_char = tf.placeholder(tf.int32, [None, None, None], name='enc_inputs_char') # size * len
        
        self.weights = tf.placeholder(tf.float32, [None, None], name='weights')
        self.tlm_weights = tf.placeholder(tf.float32, [None, None], name='tlm_weights')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.targets_reverse = tf.placeholder(tf.int32, [None, None], name='targets_reverse')
        self.tlm_targets = tf.placeholder(tf.int32, [None, None], name='tlm_targets')
        self.tlm_targets_reverse = tf.placeholder(tf.int32, [None, None], name='tlm_targets_reverse')
        self.tlm_targets_pos = tf.placeholder(tf.int32, [None, None], name='tlm_targets_pos')
        self.tlm_targets_pos_reverse = tf.placeholder(tf.int32, [None, None], name='tlm_targets_pos_reverse')
        
        self.perturb = tf.placeholder(tf.float32, [None, None], name='perturb')
    
        embedding_model = tf.get_variable('embedding', initializer=embedding_global.astype(np.float32))
        
        embedding_model_char = tf.get_variable('embedding_char', initializer=embedding_char_global.astype(np.float32))

        def delta(v):
            return tf.norm(v, ord=1)

        inputs = tf.nn.embedding_lookup(embedding_model, self.enc_inputs)
        inputs = tf.cast(inputs, tf.float32)
        
        next_inputs = tf.nn.embedding_lookup(embedding_model, self.next_enc_inputs) 
        next_inputs = tf.cast(next_inputs, tf.float32)
        # but use self.next_enc_inputs as targets in LM
        
        inputs_reverse = tf.nn.embedding_lookup(embedding_model, self.enc_inputs_reverse)
        inputs_reverse = tf.cast(inputs_reverse, tf.float32)
        
        next_inputs_reverse = tf.nn.embedding_lookup(embedding_model, self.next_enc_inputs_reverse) 
        next_inputs_reverse = tf.cast(next_inputs_reverse, tf.float32)
        
        
    
        inputs_char = tf.nn.embedding_lookup(embedding_model_char, self.enc_inputs_char)
        inputs_char = tf.cast(inputs_char, tf.float32)
        
        
        ''' Implementing TLM 
        - Tag embeddings are L dimensional one-hot vectors // why not just random initialization
        - GRU (paper uses LSTM) language model on the tag sequences
        '''
        
        with tf.variable_scope('tlm_projection'):
            proj_tlm_W = tf.get_variable('tlm_W', [dim_h, pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b = tf.get_variable('tlm_b', [pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_W_reverse = tf.get_variable('tlm_W_reverse', [dim_h, pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
            proj_tlm_b_reverse = tf.get_variable('tlm_b_reverse', [pos_size+tag_size], dtype=tf.float32) # tag_size+vocab_size
        
        
        
        

        
        
        y_onehot_tlm = tf.one_hot(self.targets, depth=tag_size) + self.perturb
        y_onehot_tlm_reverse = tf.one_hot(self.targets_reverse, depth=tag_size) + self.perturb
        
        
        inputs_pos_onehot = tf.one_hot(self.inputs_pos, depth=pos_size)
        inputs_pos_onehot_reverse = tf.one_hot(self.inputs_pos_reverse, depth=pos_size)
        inputs_chunk_onehot = tf.one_hot(self.inputs_chunk, depth=chunk_size)
        inputs_chunk_onehot_reverse = tf.one_hot(self.inputs_chunk_reverse, depth=chunk_size)
        inputs_case_onehot = tf.one_hot(self.inputs_case, depth=2) # changed from 4 to 2
        inputs_case_onehot_reverse = tf.one_hot(self.inputs_case_reverse, depth=2)
        inputs_num_onehot = tf.one_hot(self.inputs_num, depth=2)
        inputs_num_onehot_reverse = tf.one_hot(self.inputs_num_reverse, depth=2)
        

        # self.output_0_shape = tf.shape(inputs)
        # self.output_1_shape = tf.shape(y_onehot_tlm)
        # self.output_2_shape = tf.shape(inputs_pos_onehot)
        # self.output_3_shape = tf.shape(inputs_chunk_onehot)
        
        
        
        
#         with tf.variable_scope('tlm'):
#             cell_gru = create_cell(dim_h, self.dropout) # lstm actually
#             # initial_state_gru = cell_gru.zero_state(batch_size, dtype=tf.float32)
#             outputs_tlm, _ = tf.nn.dynamic_rnn(cell_gru, 
#                                                tf.concat([inputs,y_onehot_tlm,inputs_pos_onehot], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm')
#             outputs_tlm = tf.nn.dropout(outputs_tlm, self.dropout)
#             outputs_tlm = tf.reshape(outputs_tlm, [-1, dim_h])

#             self.logits_tlm_tmp = tf.matmul(outputs_tlm, proj_tlm_W) + proj_tlm_b
#             self.logits_tlm = self.logits_tlm_tmp[:,pos_size:] # FIX!!!!!!!!!
#             # self.logits_nextword = self.logits_tlm_tmp[:,:vocab_size]
#             self.logits_pos = self.logits_tlm_tmp[:,:pos_size]

#             self.probs_tlm = tf.nn.softmax(self.logits_tlm)
#             # self.probs_nextword = tf.nn.softmax(self.logits_nextword)
#             self.probs_pos = tf.nn.softmax(self.logits_pos)

        
#             loss_pretrain_tlm = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets, [-1]),
#                logits=self.logits_tlm)
#             loss_pretrain_tlm *= tf.reshape(self.tlm_weights, [-1])
# #             loss_pretrain_nextword = tf.nn.sparse_softmax_cross_entropy_with_logits(
# #                labels=tf.reshape(self.next_enc_inputs, [-1]),
# #                logits=self.logits_nextword)
# #             loss_pretrain_nextword *= tf.reshape(self.weights, [-1])
#             loss_pretrain_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_pos, [-1]),
#                logits=self.logits_pos)
#             loss_pretrain_pos *= tf.reshape(self.tlm_weights, [-1])



#         with tf.variable_scope('tlm_reverse'):
#             cell_gru_reverse = create_cell(dim_h, self.dropout)
#             outputs_tlm_reverse, _ = tf.nn.dynamic_rnn(cell_gru_reverse, 
#                                                tf.concat([inputs_reverse,y_onehot_tlm_reverse,inputs_pos_onehot_reverse], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm_reverse')
#             outputs_tlm_reverse = tf.nn.dropout(outputs_tlm_reverse, self.dropout)
#             outputs_tlm_reverse = tf.reshape(outputs_tlm_reverse, [-1, dim_h])
                    

            
#             self.logits_tlm_tmp_reverse = tf.matmul(outputs_tlm_reverse, proj_tlm_W_reverse) + proj_tlm_b_reverse
#             self.logits_tlm_reverse = self.logits_tlm_tmp_reverse[:,pos_size:] # FIX!!!!!!!!!
#             # self.logits_nextword_reverse = self.logits_tlm_tmp_reverse[:,:vocab_size]
#             self.logits_pos_reverse = self.logits_tlm_tmp_reverse[:,:pos_size]
            
#             self.probs_tlm_reverse = tf.nn.softmax(self.logits_tlm_reverse)
#             # self.probs_nextword_reverse = tf.nn.softmax(self.logits_nextword_reverse)
#             self.probs_pos_reverse = tf.nn.softmax(self.logits_pos_reverse)
            



#             loss_pretrain_tlm_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_reverse, [-1]),
#                logits=self.logits_tlm_reverse)
#             loss_pretrain_tlm_reverse *= tf.reshape(self.tlm_weights, [-1])
# #             loss_pretrain_nextword_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
# #                labels=tf.reshape(self.next_enc_inputs_reverse, [-1]),
# #                logits=self.logits_nextword_reverse)
# #             loss_pretrain_nextword_reverse *= tf.reshape(self.weights, [-1])
#             loss_pretrain_pos_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                labels=tf.reshape(self.tlm_targets_pos_reverse, [-1]),
#                logits=self.logits_pos_reverse)
#             loss_pretrain_pos_reverse *= tf.reshape(self.tlm_weights, [-1])
            

            
#         #     #self.tlm_tot_loss_0 = tf.reduce_sum(loss_pretrain_nextword)
#         #     self.tlm_tot_loss_1 = tf.reduce_sum(loss_pretrain_tlm)
#         #     self.tlm_tot_loss_2 = tf.reduce_sum(loss_pretrain_pos)
#         #     self.tlm_tot_loss = self.tlm_tot_loss_1 + self.tlm_tot_loss_2 #+ self.tlm_tot_loss_2 #
#         #     self.tlm_sent_loss_1 = self.tlm_tot_loss_1 / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_2 = self.tlm_tot_loss_2 / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss = self.tlm_tot_loss / tf.to_float(self.batch_size)
            
#         #     #self.tlm_tot_loss_0_reverse = tf.reduce_sum(loss_pretrain_nextword_reverse)
#         #     self.tlm_tot_loss_1_reverse = tf.reduce_sum(loss_pretrain_tlm_reverse)
#         #     self.tlm_tot_loss_2_reverse = tf.reduce_sum(loss_pretrain_pos_reverse)
#         #     self.tlm_tot_loss_reverse = self.tlm_tot_loss_1_reverse + self.tlm_tot_loss_2_reverse #+ self.tlm_tot_loss_2_reverse
#         #     self.tlm_sent_loss_1_reverse = self.tlm_tot_loss_1_reverse / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_2_reverse = self.tlm_tot_loss_2_reverse / tf.to_float(self.batch_size)
#         #     self.tlm_sent_loss_reverse = self.tlm_tot_loss_reverse / tf.to_float(self.batch_size)
            
            
#         # self.tlm_train_loss_1 = self.tlm_sent_loss_1+self.tlm_sent_loss_1_reverse
#         # self.tlm_train_loss_2 = self.tlm_sent_loss_2+self.tlm_sent_loss_2_reverse

        
#         # tlm_param = retrive_var(['tlm_projection','tlm','tlm_reverse'])
#         # self.optimizer_tlm_1 = tf.train.AdamOptimizer(self.learning_rate,
#         #     beta1, beta2).minimize(self.tlm_train_loss_1, var_list=tlm_param)
#         # self.optimizer_tlm_2 = tf.train.AdamOptimizer(self.learning_rate,
#         #     beta1, beta2).minimize(self.tlm_train_loss_2, var_list=tlm_param)



        
        ''' Implementing A_phi
        - An RNN that returns a vector at each position of x
        - We can interpret this vector as prob distn over output labels at that position
        - We first try an architecture of BiLSTM for A_phi
        '''
        
        with tf.variable_scope('phi_projection'):
            proj_W = tf.get_variable('W', [2*dim_h, tag_size], dtype=tf.float32) # 2 because of BiLSTM 
            proj_b = tf.get_variable('b', [tag_size], dtype=tf.float32)
        
        with tf.variable_scope('phi'):
            cell_fw = create_cell(dim_h, self.dropout)
            cell_bw = create_cell(dim_h, self.dropout)
            initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            
            
            logits_cnn = cnn(inputs_char[0],'phi') # batch size 1
            logits_cnn = tf.cast(logits_cnn, tf.float32)
            logits_cnn = tf.expand_dims(logits_cnn, 0)
            self.shape0 = tf.shape(logits_cnn) # [20,64]
            
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                tf.concat([inputs, logits_cnn, inputs_pos_onehot,inputs_chunk_onehot,inputs_case_onehot,inputs_num_onehot], axis=-1),  #inputs_pos_onehot
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32, scope='phi')
            

            
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.dropout(outputs, self.dropout)
            outputs = tf.reshape(outputs, [-1, 2*dim_h])
            outputs = tf.cast(outputs, tf.float32)
            
            self.shape1 = tf.shape(outputs) # [20,256]
            

        # affine transformation to get logits
        self.phi_logits = tf.matmul(tf.concat([outputs], axis=-1), proj_W) + proj_b # shape is (batch_size(2)*batch_length, 28)
        self.phi_probs = tf.nn.softmax(self.phi_logits) # changed from sigmoid to softmax
        # But the thing is some of the logits do not count - we need to deal with it
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
#             tf.expand_dims(self.phi_logits), self.targets, self.batch_size)

#         self.loss_crf = tf.reduce_mean(-log_likelihood)
        
        
#         labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
#         self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.viterbi_decode(
#             tf.expand_dims(self.phi_logits), transition_params)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        





        phi_probs_for_input = tf.reshape(self.phi_probs, [self.batch_size, self.batch_len, tag_size])
        phi_probs_for_input_reverse = tf.reshape(self.phi_probs[::-1,:], [self.batch_size, self.batch_len, tag_size])

#         with tf.variable_scope('tlm', reuse=True):
#             outputs_tlm_eval, _ = tf.nn.dynamic_rnn(cell_gru, 
#                                                tf.concat([inputs,phi_probs_for_input,inputs_pos_onehot], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm')
#             outputs_tlm_eval = tf.nn.dropout(outputs_tlm_eval, self.dropout)
#             outputs_tlm_eval = tf.reshape(outputs_tlm_eval, [-1, dim_h])

#             self.logits_tlm_tmp_eval = tf.matmul(outputs_tlm_eval, proj_tlm_W) + proj_tlm_b
#             self.logits_tlm_eval = self.logits_tlm_tmp_eval[:,pos_size:] # FIX!!!!!!!!!
#             self.logits_pos_eval = self.logits_tlm_tmp_eval[:,:pos_size]
            
#             self.probs_tlm_eval = tf.nn.softmax(self.logits_tlm_eval)
#             self.probs_pos_eval = tf.nn.softmax(self.logits_pos_eval)
            
#         with tf.variable_scope('tlm_reverse', reuse=True):
#             outputs_tlm_reverse_eval, _ = tf.nn.dynamic_rnn(cell_gru_reverse, 
#                                                tf.concat([inputs_reverse,phi_probs_for_input_reverse,inputs_pos_onehot_reverse], axis=-1), # [inputs,y_onehot_tlm]
#                                                dtype=tf.float32, scope='tlm_reverse')
#             outputs_tlm_reverse_eval = tf.nn.dropout(outputs_tlm_reverse_eval, self.dropout)
#             outputs_tlm_reverse_eval = tf.reshape(outputs_tlm_reverse_eval, [-1, dim_h])

            
#             self.logits_tlm_tmp_reverse_eval = tf.matmul(outputs_tlm_reverse_eval, proj_tlm_W_reverse) + proj_tlm_b_reverse
#             self.logits_tlm_reverse_eval = self.logits_tlm_tmp_reverse_eval[:,pos_size:] # FIX!!!!!!!!!
#             self.logits_pos_reverse_eval = self.logits_tlm_tmp_reverse_eval[:,:pos_size]
            
#             self.probs_tlm_reverse_eval = tf.nn.softmax(self.logits_tlm_reverse_eval)
#             self.probs_pos_reverse_eval = tf.nn.softmax(self.logits_pos_reverse_eval)







        ''' Implementing energy function '''
        
        with tf.variable_scope('energy_function'):
            energy_U = tf.get_variable('energy_U', [tag_size, dim_d+50], dtype=tf.float32)
            energy_W = tf.get_variable('energy_W', [tag_size, tag_size], dtype=tf.float32)
        
        # with tf.variable_scope('energy_feature_proj'):
        #     energy_proj_W = tf.get_variable('energy_proj_W', [2*dim_h, dim_d], dtype=tf.float32) # 2 because of BiLSTM 
        #     energy_proj_b = tf.get_variable('energy_proj_b', [dim_d], dtype=tf.float32)
        
        with tf.variable_scope('energy_feature'):
            cell_fw = create_cell(dim_h, self.dropout)
            cell_bw = create_cell(dim_h, self.dropout)
            initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32, scope='energy_feature')
            
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.dropout(outputs, self.dropout)
            outputs = tf.reshape(outputs, [-1, 2*dim_h])
            outputs = tf.cast(outputs, tf.float32)








        
        with tf.variable_scope('energy_feature_pos'):
            cell_fw_pos = create_cell(25, self.dropout)
            cell_bw_pos = create_cell(25, self.dropout)
            initial_state_fw_pos = cell_fw_pos.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw_pos = cell_bw_pos.zero_state(batch_size, dtype=tf.float32)
            outputs_pos, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_pos, cell_bw_pos, inputs_pos_onehot, 
                initial_state_fw=initial_state_fw_pos,
                initial_state_bw=initial_state_bw_pos,
                dtype=tf.float32, scope='energy_feature_pos')
            
            outputs_pos = tf.concat(outputs_pos, axis=-1)
            outputs_pos = tf.nn.dropout(outputs_pos, self.dropout)
            outputs_pos = tf.reshape(outputs_pos, [-1, 2*25])
            outputs_pos = tf.cast(outputs_pos, tf.float32)




        # shape is (batch_size(2)*batch_length, 100)
        energy_feature_vec = tf.concat([outputs,outputs_pos],axis=-1) #tf.matmul(outputs, energy_proj_W) + energy_proj_b

        # concat with pos feature vec
        # fix energy_U etc dimension 

        
        def energy_result(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse):

        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = y[:-1] # check y has 28 as last dim
            tmp1 = tf.multiply(tf.matmul(y_prime, energy_W), y[1:]) # first y is tricky
            energy_second_part = tf.reduce_sum(tmp1)
            old_return = -(energy_first_part+energy_second_part)
            
            
#             # now implement E_TLM
#             tmp_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(y[1:-1],self.probs_tlm_eval[:-2]), axis=-1))
#             tmp_energy_tlm = tf.reduce_sum(tmp_energy_tlm)
#             old_return += 0.075 * tmp_energy_tlm
        
            
#             tmp_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(y[::-1][1:-1],self.probs_tlm_reverse_eval[:-2]), axis=-1))
#             tmp_energy_tlm_reverse = tf.reduce_sum(tmp_energy_tlm_reverse)
#             old_return += 0.075 * tmp_energy_tlm_reverse
            





#             # pos_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos_eval[:-2], self.probs_pos[:-2]), axis=-1))
#             # pos_energy_tlm = tf.reduce_sum(pos_energy_tlm)
#             # old_return += 0.075 * pos_energy_tlm

#             # pos_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos_reverse_eval[:-2], self.probs_pos_reverse[:-2]), axis=-1))
#             # pos_energy_tlm_reverse = tf.reduce_sum(pos_energy_tlm_reverse)
#             # old_return += 0.075 * pos_energy_tlm_reverse

#             pos_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos_eval[:-2], self.probs_pos_reverse_eval[:-2][::-1]), axis=-1))
#             pos_energy_tlm = tf.reduce_sum(pos_energy_tlm)
#             old_return += 0.10 * pos_energy_tlm





            return old_return



        def energy_result_gold(self, x, y, y_unscale_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse):

        
            # note that energy_feature_vec will be looped around twice with batch_size 2
            M0 = tf.matmul(energy_U, tf.transpose(energy_feature_vec)) 
            tmp0 = tf.multiply(y, tf.transpose(M0)) # elt-wise
            energy_first_part = tf.reduce_sum(tmp0)
            
            #y_prime = tf.manip.roll(y, shift=1, axis=0)
            #y_prime = tf.concat([[tf.zeros([tag_size])], y_prime[1:]], axis=0) # check y has 28 as last dim
            
            y_prime = y[:-1] # check y has 28 as last dim
            tmp1 = tf.multiply(tf.matmul(y_prime, energy_W), y[1:]) # first y is tricky
            energy_second_part = tf.reduce_sum(tmp1)
            old_return = -(energy_first_part+energy_second_part)
            
            
#             # now implement E_TLM
#             tmp_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(y[1:-1],self.probs_tlm[:-2]), axis=-1))
#             tmp_energy_tlm = tf.reduce_sum(tmp_energy_tlm)
#             old_return += 0.075 * tmp_energy_tlm
        
            
#             tmp_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(y[::-1][1:-1],self.probs_tlm_reverse[:-2]), axis=-1))
#             tmp_energy_tlm_reverse = tf.reduce_sum(tmp_energy_tlm_reverse)
#             old_return += 0.075 * tmp_energy_tlm_reverse
            






#             # pos_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos[:-2], self.probs_pos[:-2]), axis=-1))
#             # pos_energy_tlm = tf.reduce_sum(pos_energy_tlm)
#             # old_return += 0.05 * pos_energy_tlm

#             # pos_energy_tlm_reverse = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos_reverse[:-2], self.probs_pos_reverse[:-2]), axis=-1))
#             # pos_energy_tlm_reverse = tf.reduce_sum(pos_energy_tlm_reverse)
#             # old_return += 0.05 * pos_energy_tlm_reverse


#             pos_energy_tlm = -tf.log(tf.reduce_sum(tf.multiply(self.probs_pos[:-2], self.probs_pos_reverse[:-2][::-1]), axis=-1))
#             pos_energy_tlm = tf.reduce_sum(pos_energy_tlm)
#             old_return += 0.10 * pos_energy_tlm




            return old_return

        
        
        ''' Implementing phi and theta '''
        
        y_onehot = tf.one_hot(self.targets, depth=tag_size)
        y_onehot = tf.reshape(y_onehot, [-1, tag_size])
        tmp_delta_0 = tf.reduce_sum(self.phi_probs - y_onehot, axis=-1)
        tmp_delta_0 *= tf.reshape(self.weights,[-1])
        
        x_nextword_onehot = tf.one_hot(self.next_enc_inputs, depth=vocab_size)
        x_nextword_onehot = tf.reshape(x_nextword_onehot, [-1, vocab_size])
        
        x_nextword_onehot_reverse = tf.one_hot(self.next_enc_inputs_reverse, depth=vocab_size)
        x_nextword_onehot_reverse = tf.reshape(x_nextword_onehot_reverse, [-1, vocab_size])
        
        nextpos_onehot = tf.one_hot(self.tlm_targets_pos, depth=pos_size)
        nextpos_onehot = tf.reshape(nextpos_onehot, [-1, pos_size])
        
        nextpos_onehot_reverse = tf.one_hot(self.tlm_targets_pos_reverse, depth=pos_size)
        nextpos_onehot_reverse = tf.reshape(nextpos_onehot_reverse, [-1, pos_size])
        
        
        extra_reg_term = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]),logits=self.phi_logits)
        extra_reg_term *= tf.reshape(self.weights, [-1])
        extra_reg_term = tf.reduce_sum(extra_reg_term) / tf.to_float(self.batch_size)



        
        
        
        
        # self.loss_phi *= tf.reshape(self.weights, [-1])
        # something like this 
        loss_phi = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) #+ energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
        loss_phi = -loss_phi
        self.loss_phi = extra_reg_term #loss_phi + 0.5 * extra_reg_term #tf.maximum(loss_phi, 0.0) + 0.5 * extra_reg_term
        
        
        lambda_new = 1.0
        new_theta_term = lambda_new * (- energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) \
                                       + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse))
        new_theta_term = tf.maximum(new_theta_term, -1.0)
        
        loss_theta = delta(tmp_delta_0) - energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse) \
            + energy_result_gold(self, inputs, y_onehot, y_onehot, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
            # + 0.0001 * retrive_var_regularize(['energy_function','energy_feature_proj','energy_feature']) # regularization
        loss_theta = tf.maximum(loss_theta, -1.0)
        self.loss_theta = loss_theta + new_theta_term
        
        
        
        
        
        
        
        ''' Optimization '''
        
        phi = retrive_var(['phi_projection','phi','embedding_char'])
        theta = retrive_var(['energy_function','energy_feature','energy_feature_pos'])#,'tlm_projection','tlm','tlm_reverse'])
        self.optimizer_phi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_phi, var_list=phi)
        self.optimizer_theta = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_theta, var_list=theta)
        
        
        psi = retrive_var(['phi_projection','phi'])
        self.loss_psi = energy_result(self, inputs, self.phi_probs, self.phi_logits, x_nextword_onehot, x_nextword_onehot_reverse, nextpos_onehot, nextpos_onehot_reverse)
        self.optimizer_psi = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_psi, var_list=psi)
        

        
        
        
        
        self.saver = tf.train.Saver()
        
        





def create_model_infnet_tlm(sess, dim_h, n_tag, n_pos, n_chunk, vocab_size, load_model=False, model_path=''):
    model = InfNet_TLM(dim_h, n_tag, n_pos, n_chunk, vocab_size)
    if load_model:
        print('Loading model from ...')
        model.saver.restore(sess, model_path)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model


def evaluate_tlm(sess, model, x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size)
    tot_loss_0, tot_loss_1, tot_loss_2, tot_loss_0_reverse, tot_loss_1_reverse, tot_loss_2_reverse, n_words = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for batch in batches:
        if batch['size'] == batch_size:
            tmp_tot_loss_1, tmp_tot_loss_2, tmp_tot_loss_1_reverse, tmp_tot_loss_2_reverse = sess.run([model.tlm_tot_loss_1, model.tlm_tot_loss_2,
                model.tlm_tot_loss_1_reverse, model.tlm_tot_loss_2_reverse],
                feed_dict={model.batch_size: batch['size'],
                           model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_char: batch['enc_inputs_char'],
                           model.enc_inputs_reverse: batch['enc_inputs_reverse'],
                           model.next_enc_inputs: batch['next_enc_inputs'],
                           model.next_enc_inputs_reverse: batch['next_enc_inputs_reverse'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_pos_reverse: batch['inputs_pos_reverse'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_chunk_reverse: batch['inputs_chunk_reverse'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_case_reverse: batch['inputs_case_reverse'],
                           model.inputs_num: batch['inputs_num'],
                           model.inputs_num_reverse: batch['inputs_num_reverse'],
                           model.batch_len: batch['len'],
                           model.targets: batch['targets'],
                           model.targets_reverse: batch['targets_reverse'],
                           model.tlm_targets: batch['tlm_targets'],
                           model.tlm_targets_reverse: batch['tlm_targets_reverse'],
                           model.tlm_targets_pos: batch['tlm_targets_pos'],
                           model.tlm_targets_pos_reverse: batch['tlm_targets_pos_reverse'],
                           model.weights: batch['weights'],
                           model.tlm_weights: batch['tlm_weights'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1})
            #tot_loss_0 += tmp_tot_loss_0
            tot_loss_1 += tmp_tot_loss_1
            tot_loss_2 += tmp_tot_loss_2
            #tot_loss_0_reverse += tmp_tot_loss_0_reverse
            tot_loss_1_reverse += tmp_tot_loss_1_reverse
            tot_loss_2_reverse += tmp_tot_loss_2_reverse
            
            n_words += np.sum(batch['weights'])

    return np.exp(tot_loss_1 / n_words), np.exp(tot_loss_2 / n_words),  np.exp(tot_loss_1_reverse / n_words), np.exp(tot_loss_2_reverse / n_words)#, np.exp(tot_loss_2_reverse / n_words)



def evaluate(sess, model, x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size)
    
    y = []
    
    same = 0
    ttl = 0
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_char: batch['enc_inputs_char'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i] and (y[i] != tag2id['<s>']) and (y[i] != tag2id['</s>']):
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs















def evaluate_print(sess, model, x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    acc_y = []
    acc_y_hat = []
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i]:
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
                    
            acc_y.append(y)
            acc_y_hat.append(y_hat)
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs, batches, acc_y, acc_y_hat













def evaluate_print(sess, model, x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size):
    probs = []
    batches, _ = get_batches(x, y, pos, chunk, case, num, word2id, tag2id, pos2id, chunk2id, batch_size)
    y = []
    
    same = 0
    ttl = 0
    
    acc_y = []
    acc_y_hat = []
    
    for batch in batches:
        if batch['size'] == batch_size:
            probs = sess.run(model.phi_probs,
                feed_dict={model.enc_inputs: batch['enc_inputs'],
                           model.enc_inputs_char: batch['enc_inputs_char'],
                           model.inputs_pos: batch['inputs_pos'],
                           model.inputs_chunk: batch['inputs_chunk'],
                           model.inputs_case: batch['inputs_case'],
                           model.inputs_num: batch['inputs_num'],
                           model.batch_len: batch['len'],
                           model.batch_size: batch['size'],
                           model.perturb: batch['perturb'],
                           model.dropout: 1.0})

            # shape is (batch_size(4), batch_length, 28)
            probs = probs.reshape((batch['size'],batch['len'],len(tag2id)))

            # shape is (batch_size(4)*batch_length)
            wt = np.array(batch['weights'])
            wt = wt.reshape(batch['size']*batch['len'])

            y = np.array(batch['targets'])
            y = y.reshape(batch['size']*batch['len'])

            y_hat = [np.argmax(p) for i in range(batch['size']) for p in probs[i]]

            for i in range(len(wt)):
                if wt[i] and (y[i] != tag2id['<s>']) and (y[i] != tag2id['</s>']):
                    if y[i] == y_hat[i]:
                        same += 1
                    ttl += 1
                    
            acc_y.append(y)
            acc_y_hat.append(y_hat)
        
#         y.append(batch['targets'][0])
#         probs += p.tolist()
#     y_hat = 
#     same = [p == q for p, q in zip(y, y_hat)]

    return 100.0 * (same) / ttl, probs, batches, acc_y, acc_y_hat














'''

# STEP 1 - TRAINING TAG LANGUAGE MODEL

NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
config = tf.ConfigProto(
    intra_op_parallelism_threads=NUM_THREADS,
    inter_op_parallelism_threads=NUM_THREADS)
config.gpu_options.allow_growth = True

print('configuring GPU')

# InfNet

steps_per_checkpoint = 1500
learning_rate = 0.0006
max_epochs = 1500
batch_size = 1
dropout = 0.7
load_model = False # True
model_name = './tmp/08-14-18-ner-series-ppf'


with tf.Graph().as_default():
    
    with tf.Session(config=config) as sess:
        

        # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
        model = create_model_infnet_tlm(sess, 128, len(tag2id), len(pos2id), len(chunk2id), len(word2id), load_model, model_name) # use 100
        
        if True: # training
            batches, _ = get_batches(x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id, batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = -1
            loss_tlm = 0.0
            best_dev = float('inf')
            #best_dev = 96.00

            for epoch in range(max_epochs):
                print('----------------------------------------------------')
                print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                for batch in batches: # note
                    
                    if batch['size'] == batch_size:
                    
                        feed_dict_tmp = feed_dictionary(model, batch, dropout, learning_rate)
#                         tmp0, tmp1 = sess.run([model.output_0_shape, model.output_1_shape],
#                            feed_dict=feed_dict_tmp)
#                         print(tmp0, tmp1)

                        step_loss_tlm_1, _ = sess.run([model.tlm_train_loss_1, model.optimizer_tlm_1],
                            feed_dict=feed_dict_tmp)
                        step_loss_tlm_2, _ = sess.run([model.tlm_train_loss_2, model.optimizer_tlm_2],
                            feed_dict=feed_dict_tmp)
                        step_loss_tlm = step_loss_tlm_1+step_loss_tlm_2

                        step += 1
                        loss_tlm += step_loss_tlm / steps_per_checkpoint

                        if step % steps_per_checkpoint == 0:
                            print('step %d, time %.0fs, loss_tlm %.2f' \
                                % (step, time.time() - start_time, loss_tlm))
                            loss_tlm = 0.0
                
                            #acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                            #print('-- dev acc: %.2f' % acc)
            
                        if step % (2*steps_per_checkpoint) == 0:
                            pp1, pp2, pp1_reverse, pp2_reverse = evaluate_tlm(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                            print('-- dev perplexity: %.2f, %.2f, %.2f, %.2f' % (pp1,pp2,pp1_reverse,pp2_reverse))
                            
                            if (3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse) < best_dev:
                                best_dev = 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse
                                print('------ best dev perplexity so far -> saving model...')
                                model.saver.save(sess, model_name)
                            
                #pp, _ = evaluate_tlm(sess, model, x_train, y_train, pos_train, chunk_train, word2id, tag2id, pos2id, chunk2id,  batch_size)
                #print('-- train pp: %.2f' % pp)

                pp1, pp2, p1_reverse, pp2_reverse = evaluate_tlm(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                print('-- dev perplexity: %.2f, %.2f, %.2f, %.2f' % (pp1,pp2,pp1_reverse,pp2_reverse))
                
                
                

# need to change dev to tweet3m_test_sent and tweet3m_test_tag
                
                if 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse < best_dev:
                    best_dev = 3*pp1+1*pp2 + 3*pp1_reverse+1*pp2_reverse
                    print('------ best dev pp so far -> saving model...')
                    model.saver.save(sess, model_name)
                
#                 # acc, _ = evaluate(sess, model, test_data, window)
#                 # print('-- test acc: %.2f' % acc)

'''













# NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
# config = tf.ConfigProto(
#     intra_op_parallelism_threads=NUM_THREADS,
#     inter_op_parallelism_threads=NUM_THREADS)
# config.gpu_options.allow_growth = True

# print('configuring GPU')


# STEP 2 - TRAINING PHI AND THETA

# InfNet

steps_per_checkpoint = 9000
learning_rate = 0.000
max_epochs = 150
batch_size = 1
dropout = 0.7
load_model = False # True
model_name = './tmp/08-14-18-ner-series-lstm-char'

with tf.Graph().as_default():
    with tf.Session() as sess:

        # model = create_model_infnet(sess, 100, len(tag2id)) # use 100
        model = create_model_infnet_tlm(sess, 128, len(tag2id), len(pos2id), len(chunk2id), len(word2id), load_model, model_name) # create right model!
        
        if True: # training
            batches, _ = get_batches(x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id, batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = -1
            loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
            #best_dev = float('-inf')
            best_dev = 95.10 # do not save

            for epoch in range(max_epochs):
                print('----------------------------------------------------')
                print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                for batch in batches:
                    
                    if batch['size'] == batch_size:
                                                
                        feed_dict_tmp = feed_dictionary(model, batch, dropout, learning_rate)
                        
#                         ### debug
#                         tmp0, tmp1 = sess.run([model.shape0, model.shape1],
#                            feed_dict=feed_dict_tmp)
#                         print(tmp0, tmp1)
#                         ###
                        

                        step_loss_phi, _ = sess.run([model.loss_phi, model.optimizer_phi],
                            feed_dict=feed_dict_tmp)
#                         step_loss_psi, _ = sess.run([model.loss_psi, model.optimizer_psi],
#                             feed_dict=feed_dict_tmp)
#                         step_loss_theta, _ = sess.run([model.loss_theta, model.optimizer_theta],
#                             feed_dict=feed_dict_tmp)

                        step += 1
                        loss_phi += step_loss_phi / steps_per_checkpoint
                        loss_psi += 0 #step_loss_psi / steps_per_checkpoint
                        loss_theta += 0 #step_loss_theta / steps_per_checkpoint

                        if step % steps_per_checkpoint == 0:
                            print('step %d, time %.0fs, loss_phi %.2f, loss_psi %.2f, loss_theta %.2f' \
                                % (step, time.time() - start_time, loss_phi, loss_psi, loss_theta))
                            loss_phi, loss_psi, loss_theta = 0.0, 0.0, 0.0
                            
                
#                print('------ ... -> saving model...')
#                model.saver.save(sess, model_name)
                
                            #acc, _ = evaluate(sess, model, x_dev, y_dev, word2id, tag2id, batch_size)
                            #print('-- dev acc: %.2f' % acc)
            
                        if step % (2*steps_per_checkpoint) == 0: # MODIFY LATER
                            acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                            print('-- dev acc: %.2f' % acc)






                            acc, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id, batch_size)
                            #acc, probs_test, batches_test, acc_y_test, acc_y_hat_test = evaluate_print(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id, batch_size)
                            #print('-- dev acc: %.2f' % acc)

                            store_lst = []
                            for bn in range(len(batches_test)):
                                batch = batches_test[bn]
                                for i in range(batch['len']):
                                    store_word_id = batch['enc_inputs'][0][i]
                                    if store_word_id not in [1,2]:
                                        store_word = id2word[store_word_id]
                                        store_pos = id2pos[batch['inputs_pos'][0][i]]
                                        store_real_tag = id2tag[acc_y_test[bn][i]]
                                        store_predicted_tag = id2tag[acc_y_hat_test[bn][i]]
                                        store_lst.append([store_word,store_pos,store_real_tag,store_predicted_tag])
                                store_lst.append([])
      
                            write_file_name = 'ner_eval_outputs.txt'
                            with open(write_file_name, 'w') as f:
                                for x in store_lst:
                                    if len(x) == 0:
                                        f.write('\n')
                                    else:
                                        assert len(x) == 4
                                        write_str = x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + x[3]
                                        f.write(write_str+'\n')
                                        
                            bash_command = 'perl conlleval < ' + write_file_name + ' > bash_result.out'
                            output = subprocess.check_output(['bash','-c', bash_command])
                            with open('bash_result.out') as f:
                                tmp = f.readlines()
                                print("F1 test:")
                                print(float(tmp[1][-6:-1]))










                            
                            if acc > best_dev:
                                best_dev = acc
                                print('------ best dev acc so far -> saving model...')
                                model.saver.save(sess, model_name)
                                
                                acc, _ = evaluate(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id,  batch_size)
                                print('-- test acc: %.2f' % acc)
                                
                            
                acc, _ = evaluate(sess, model, x_train, y_train, pos_train, chunk_train, case_train, num_train, word2id, tag2id, pos2id, chunk2id,  batch_size)
                print('-- train acc: %.2f' % acc)

                acc, _ = evaluate(sess, model, x_dev, y_dev, pos_dev, chunk_dev, case_dev, num_dev, word2id, tag2id, pos2id, chunk2id,  batch_size)
                print('-- dev acc: %.2f' % acc)
                

                if acc > best_dev:
                    best_dev = acc
                    print('------ best dev acc so far -> saving model...')
                    model.saver.save(sess, model_name)
                    
                    acc, _ = evaluate(sess, model, x_test, y_test, pos_test, chunk_test, case_test, num_test, word2id, tag2id, pos2id, chunk2id,  batch_size)
                    print('-- test acc: %.2f' % acc)
                

                if int(time.time()-start_time) > 13000:
                    print('=== saving model after 13000 seconds -> saving')
                    model.saver.save(sess, model_name+'-contd')






