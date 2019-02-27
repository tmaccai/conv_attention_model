import multiprocessing
import random
import sys
import jieba
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import word2vec
from tensorflow.contrib import rnn


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


# keep chinese only and tokenization
def tokenization(s):
    content_str = ''
    for i in s:
        if is_chinese(i):
            content_str = content_str + i
    l = jieba.lcut(content_str)
    if len(l) == 0:
        return None
    else:
        return l


# word2vec embedding, [https://blog.csdn.net/guyuealian/article/details/84072158]
def train_wordVectors(sentences, embedding_size=128, window=5, min_count=5):
    w2vModel = word2vec.Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                                 workers=multiprocessing.cpu_count())
    return w2vModel


# covert a word to a vector
def word_to_vec(word, model):
    try:
        return model.get_vector(word)
    # if unseen, return a zero vector
    except:
        return np.zeros(model.vector_size)


# convert any string into a sequence of vectors
# output a list with fixed length T
def get_seq(s, model, max_len):
    l = tokenization(s)
    seq = [word_to_vec(w, model) for w in l]
    n = len(seq)
    if n > max_len:
        return np.asarray(seq[:max_len])
    else:
        # padding with zero vectors
        output = seq + [[0] * model.vector_size] * (max_len - n)
        return np.asarray(output)


def one_hot(label):
    if label == 0:
        return np.array([[0], [1]])
    else:
        return np.array([[1], [0]])


def prediction(y_pred):
    if y_pred <= 0.5:
        return 'neg'
    else:
        return 'pos'


def main():

    # load txt into data frame
    raw_neg = pd.read_csv("neg.txt", sep="\\n", header=None)
    raw_pos = pd.read_csv("pos.txt", sep="\\n", header=None)

    df_neg, df_pos = raw_neg.applymap(tokenization), raw_pos.applymap(tokenization)
    df_neg, df_pos = df_neg.dropna(axis=0), df_pos.dropna(axis=0)
    df_neg.reset_index(drop=True), df_pos.reset_index(drop=True)
    n_neg, n_pos = df_neg.shape[0], df_pos.shape[0]
    # add labels, 0 for negative and 1 for positive
    df_neg[1], df_pos[1] = 0.0, 1.0

    df_total = pd.concat([df_neg, df_pos], axis=0)
    df_total.reset_index(drop=True)
    n_total = df_total.shape[0]

    # hyper parameters
    l, d, T, n_filter, batch_size = 3, 128, 20, 100, 32

    # list of all sentences, each element is a list of words
    l_sen = df_total[0].tolist()

    word2vec_path = './word2Vec.model'
    w2v_model = train_wordVectors(l_sen, embedding_size=d, window=5, min_count=5)
    word_vec = w2v_model.wv

    tf.reset_default_graph()

    # CNN with SGD
    # true label
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    # input a T*d 2d array
    x = tf.placeholder(dtype=tf.float32, shape=[None, T, d])

    ####################################
    ## RNN, [https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py]
    learning_rate = 0.04

    # Network Parameters
    num_input = d  # data input
    timesteps = T  # timesteps
    num_hidden = d  # hidden layer num of features
    num_classes = 2  # total classes

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    x_lstm = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output, outputs [T, batch_size, d], states [num_classes, batch_size, d]
    outputs, states = rnn.static_rnn(lstm_cell, x_lstm, dtype=tf.float32)
    rnn_features = tf.transpose(outputs, [1, 0, 2])

    ####################################

    # input_layer = tf.reshape(x_input, [-1, T, d, 1])
    x_input = tf.reshape(x, [-1, T, d, 1])
    # conv1 shape
    conv1 = tf.layers.conv2d(inputs=x_input,
                             filters=n_filter,
                             strides=[1, d],
                             kernel_size=[l, d],
                             padding="same",
                             activation=tf.nn.leaky_relu)

    # attention signal, 1d array with shape [batch_size, T, 1]
    attention = tf.reduce_mean(conv1, 3)
    # dropout, shape [batch_size, T, 1]
    attention = tf.nn.dropout(attention, 0.5)
    # attention = tf.layers.dropout(attention, 0.5)
    weighted_x = tf.multiply(rnn_features, attention)
    # x for classification, shape [batch_size, d]
    out_x = tf.reduce_mean(weighted_x, 1)
    # logistic layer, shape [batch_size, 1]
    logits = tf.layers.dense(inputs=out_x, units=1, use_bias=True)
    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true))
    y_pred = tf.math.sigmoid(logits)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    random.seed(1)


    # indices = list(range(n_total))
    # random.shuffle(indices)

    # temporary loss and max number of iterations

    def training():
        pre_loss, max_iter = 100, 40000
        with tf.Session() as sess:
            print('Training...')
            sess.run(init)
            # Train
            for _ in range(max_iter):
                l_ind = random.sample(list(range(n_total)), batch_size)
                df_batch = df_total.iloc[l_ind, :]
                x_batch = df_batch[0].apply(lambda s: get_seq(s, word_vec, T))
                # reshape to [batch_size, T, d]
                x_batch = np.asarray(list(x_batch))
                y_batch = df_batch[1]
                # reshape to [batch_size, 2, 1]]
                y_batch = np.asarray(y_batch).reshape(batch_size, 1)
                cur_loss = sess.run(loss, {x: x_batch, y_true: y_batch})
                #if abs(cur_loss - pre_loss) > 0.000001:
                sess.run(train_step, {x: x_batch, y_true: y_batch})
                pre_loss = cur_loss
                #else:
                 #   break

            # save model
            save_path = saver.save(sess, "./cran_model")
        print("Model saved in path: %s" % save_path)


    def testing(input_file):
        with tf.Session() as sess:
            # load model
            saver.restore(sess, "./cran_model")
            # Prediction
            df_test = pd.read_csv(input_file, sep="\\n", header=None)
            df_test[1] = df_test[0].map(tokenization)
            df_test = df_test.dropna()
            x_test = df_test[1].apply(lambda s: get_seq(s, word_vec, T))
            x_test = np.asarray(list(x_test))

            # extract key feature
            n = df_test.shape[0]
            temp_attention = sess.run(attention, {x: x_test})
            sen_len = list(df_test[1].apply(len))
            l_features = []
            for i in range(n):
                temp_s = temp_attention[i, :sen_len[i], :]
                k = np.argmax(temp_s)
                try:
                    best_word = df_test.iloc[i, 1][k]
                    l_features.append(best_word)
                except:
                    l_features.append('')

            # predict
            print('Predicting...')
            y_out = pd.DataFrame(sess.run(y_pred, {x: x_test}))
            y_out = pd.Series(y_out[0].apply(prediction))
            df_out = pd.concat([df_test[0].reset_index(drop=True), y_out.reset_index(drop=True)], axis=1)
            df_out[2] = l_features
            df_out.columns = ['comments', 'labels', 'key_feature']
            df_out.to_csv('prediction.txt', index=False)
            print('Output to prediction.txt')

    mode = sys.argv[1]
    if mode == 'train':
        training()
    elif mode == 'test':
        testing(sys.argv[2])
    else:
        print('wrong command')

if __name__ == '__main__':
    main()
