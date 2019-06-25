import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import numpy as np
import os
#import tqdm

in_seq = []
out_seq = []
with open('train/in.txt',encoding='UTF-8') as f:
    allLines = f.readlines()
    allLines = [item.strip() for item in allLines]
    for item in allLines:
        tmpSentence = item.split(' ')
        tmpSentence = [one.strip() for one in tmpSentence if one.strip() != '']
        in_seq.append(tmpSentence)

with open('train/out.txt',encoding='UTF-8') as f:
    allLines = f.readlines()
    allLines = [item.strip() for item in allLines]
    for item in allLines:
        tmpSentence = item.split(' ')
        tmpSentence = [one.strip() for one in tmpSentence if one.strip() != '']
        out_seq.append(tmpSentence)

vocabs = []
with open('train/vocabs',encoding='UTF-8') as f:
    vocabs.extend(f.readlines())
    vocabs = [item.strip() for item in vocabs]

vocab_to_int = dict(zip(vocabs,range(len(vocabs))))
int_to_vocab = dict(zip(range(len(vocabs)),vocabs))

print(len(in_seq))
print(len(out_seq))

data = []
for i in range(len(in_seq)):
    one_in_seq = in_seq[i]
    one_out_seq = out_seq[i]
    one_in_seq = [vocab_to_int[one] for one in one_in_seq]
    one_in_seq = one_in_seq+[vocab_to_int['</s>']]
    one_out_seq = [vocab_to_int[one] for one in one_out_seq]
    one_out_seq = [vocab_to_int['<s>']]+one_out_seq+[vocab_to_int['</s>']]
    data.append({
        'in_seq':one_in_seq,
        'in_seq_len':len(one_in_seq),
        'out_seq':one_out_seq,
        'out_seq_len':len(one_out_seq) - 1
    })

def padding_seq(seq):
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results

def get_Batch(batch_size):
    num_batch = ((len(data) - 10) // batch_size) +1
    for i in range(num_batch):
        batch_data = data[i*batch_size:min((i+1)*batch_size,len(data))]
        batch = {'in_seq': [],
                 'in_seq_len': [],
                 'out_seq': [],
                 'out_seq_len': []}
        for item in batch_data:
            batch['in_seq'].append(item['in_seq'])
            batch['in_seq_len'].append(item['in_seq_len'])
            batch['out_seq'].append(item['out_seq'])
            batch['out_seq_len'].append(item['out_seq_len'])
        batch['in_seq'] = padding_seq(batch['in_seq'])
        batch['out_seq'] = padding_seq(batch['out_seq'])
        yield batch

def getLayeredCell(layer_size, num_units, input_keep_prob,
        output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units),
        input_keep_prob, output_keep_prob) for i in range(layer_size)])


class CoupletConfig(object):

    embedding_dim = 300
    vocab_size = len(vocabs)

    num_units = 1024
    layer_size = 4

    keep_prob = 0.8
    learning_rate = 1e-3
    batch_size = 50


class CoupletModel(object):
    def __init__(self,config):
        self.config = config
        self.x_input = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.y_input = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.x_input_len = tf.placeholder(dtype=tf.int32,shape=[None])
        self.y_input_len = tf.placeholder(dtype=tf.int32,shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.embedding = tf.get_variable('embedding',shape=[self.config.vocab_size,self.config.embedding_dim])

        self.embed_input = tf.nn.embedding_lookup(params=self.embedding,ids=self.x_input)
        self.embed_output = tf.nn.embedding_lookup(params=self.embedding,ids=self.y_input)

        self.encoder_output,self.encoder_state = self.encoder_layer()
        self.attention_cell = self.attention_decoder_cell()
        self.projection_layer = tf.layers.Dense(self.config.vocab_size,use_bias=False)
        self.batch_size = tf.shape(self.x_input_len)[0]
        self.init_state = self.attention_cell.zero_state(self.batch_size, tf.float32).clone(
            cell_state=self.encoder_state)

        self.train_helper = seq2seq.TrainingHelper(inputs=self.embed_output,sequence_length=self.y_input_len,time_major=False)
        self.infer_helper = seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,start_tokens=tf.fill([self.batch_size], 0),end_token=1)

        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.attention_cell, self.train_helper,
            self.init_state, output_layer=self.projection_layer)
        self.infer_decoder = tf.contrib.seq2seq.BasicDecoder(self.attention_cell, self.infer_helper,
            self.init_state, output_layer=self.projection_layer)

        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder,maximum_iterations=100)
        infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(self.infer_decoder,maximum_iterations=100)

        self.train_outputs = train_outputs.rnn_output
        self.infer_outputs = infer_outputs.sample_id

        masks = tf.sequence_mask(self.y_input_len, tf.shape(self.train_outputs)[1], dtype=tf.float32, name="masks")
        self.cost = tf.contrib.seq2seq.sequence_loss(self.train_outputs, self.y_input[:, 1:], masks)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        gradients = optimizer.compute_gradients(self.cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gradients)

        self.train()





    def encoder_layer(self):
        bi_layer_size = int(self.config.layer_size / 2)
        encode_cell_fw = getLayeredCell(bi_layer_size, self.config.num_units, self.keep_prob)
        encode_cell_bw = getLayeredCell(bi_layer_size, self.config.num_units, self.keep_prob)
        bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=encode_cell_fw,
            cell_bw=encode_cell_bw,
            inputs=self.embed_input,
            sequence_length=self.x_input_len,
            dtype=self.embed_input.dtype,
            time_major=False)

        # concat encode output and state
        encoder_output = tf.concat(bi_encoder_output, -1)
        encoder_state = []
        for layer_id in range(bi_layer_size):
            encoder_state.append(bi_encoder_state[0][layer_id])
            encoder_state.append(bi_encoder_state[1][layer_id])
        encoder_state = tuple(encoder_state)
        return encoder_output, encoder_state

    def attention_decoder_cell(self):
        attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(self.config.num_units,
                                    self.encoder_output, self.x_input_len, normalize=True)
        # attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,
        #         encoder_output, in_seq_len, scale = True)
        cell = getLayeredCell(self.config.layer_size, self.config.num_units, self.keep_prob)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
                                                   attention_layer_size=self.config.num_units)
        return cell

    def train(self):
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # self.saver.restore(sess, 'model/best_model')
            min_loss = 100
            for i in range(1000):
                for j,one_batch in enumerate(get_Batch(self.config.batch_size)):
                    loss_val,_ = sess.run([self.cost,self.train_op],feed_dict={
                        self.x_input : one_batch['in_seq'],
                        self.y_input : one_batch['out_seq'],
                        self.x_input_len : one_batch['in_seq_len'],
                        self.y_input_len : one_batch['out_seq_len'],
                        self.keep_prob : self.config.keep_prob
                    })
                    print(loss_val)
                    if j % 101 == 0:
                        self.saver.save(sess,'modelBeifen/beifen_model')
                    if j% 10 == 0:
                        if loss_val < min_loss:
                            self.saver.save(sess,'model/best_model')
                            min_loss = loss_val
                        infer_outputs_val = sess.run(self.infer_outputs, feed_dict={
                            self.x_input: one_batch['in_seq'],
                            self.y_input: one_batch['out_seq'],
                            self.x_input_len: one_batch['in_seq_len'],
                            self.y_input_len: one_batch['out_seq_len'],
                            self.keep_prob: 1.0
                        })
                        for item in infer_outputs_val:
                            item = [int_to_vocab[one] for one in item]
                            print(item)
config = CoupletConfig()
model = CoupletModel(config)
