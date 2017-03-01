#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy.io
import glob
import os
import tensorflow as tf

import sys
sys.path.append('./tensorflow_hmm')
import tensorflow_hmm.hmm as hmm

class DataLoader:
    def __init__(self):

        self.batch_idx = 0

        self.index_name = ['end', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                      'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

        # load the data
        pose_pre_data_all = []
        pose_post_data_all = []
        force_pre_data_all = []
        force_post_data_all = []
        next_action_label_all = []
        current_action_label_all = []

        mat_files = glob.glob('data/*.mat')
        for mat_file in mat_files:
            data = scipy.io.loadmat(mat_file)
            pose_num = data['pose_window_width'][0][0]
            force_num = data['force_window_width'][0][0]
            data = data['windows']

            pose_pre_data = np.array(data[0::2, 3:3+pose_num])
            pose_post_data = np.array(data[1::2, 3:3+pose_num])
            force_pre_data = np.array(data[0::2, 3+pose_num:3+pose_num+force_num])
            force_post_data = np.array(data[1::2, 3+pose_num:3+pose_num+force_num])
            next_action_label = np.array([max(0, x) for x in data[0::2, -1]])

            current_action_label = np.repeat(self.index_name.index(mat_file.split('/')[-1][:-12]), pose_pre_data.shape[0])

            pose_pre_data_all.extend(pose_pre_data)
            pose_post_data_all.extend(pose_post_data)

            force_pre_data_all.extend(force_pre_data)
            force_post_data_all.extend(force_post_data)
            next_action_label_all.extend(next_action_label)
            current_action_label_all.extend(current_action_label)

        pose_pre_data_all = np.array(pose_pre_data_all)
        pose_post_data_all = np.array(pose_post_data_all)
        force_pre_data_all = np.array(force_pre_data_all)
        force_post_data_all = np.array(force_post_data_all)
        next_action_label_all = np.array(next_action_label_all, dtype=np.uint8)
        current_action_label_all = np.array(current_action_label_all, dtype=np.uint8)

        # shuffle the data
        shuffle_index = np.arange(pose_pre_data_all.shape[0])
        np.random.shuffle(shuffle_index)
        pose_pre_data_all = pose_pre_data_all[shuffle_index, :]
        pose_post_data_all = pose_post_data_all[shuffle_index, :]
        force_pre_data_all = force_pre_data_all[shuffle_index, :]
        force_post_data_all = force_post_data_all[shuffle_index, :]

        next_action_label_all = next_action_label_all[shuffle_index]
        next_action_label_vec = self.one_hot(next_action_label_all, len(self.index_name))
        current_action_label_all = current_action_label_all[shuffle_index]
        current_action_label_vec = self.one_hot(current_action_label_all, len(self.index_name))

        # split the data
        num_training = int(pose_pre_data_all.shape[0] * 0.8)
        training_pre_data = np.hstack((pose_pre_data_all[:num_training, :], force_pre_data_all[:num_training, :]))
        training_post_data = np.hstack((pose_post_data_all[:num_training, :], force_post_data_all[:num_training, :]))
        training_current_action = current_action_label_vec[:num_training, :]
        training_next_action = next_action_label_vec[:num_training, :]

        testing_pre_data = np.hstack((pose_pre_data_all[num_training:, :], force_pre_data_all[num_training:, :]))
        testing_post_data = np.hstack((pose_post_data_all[num_training:, :], force_post_data_all[num_training:, :]))
        testing_current_action = current_action_label_vec[num_training:, :]
        testing_next_action = next_action_label_vec[num_training:, :]

        self.training_pre_data = training_pre_data
        self.training_post_data = training_post_data
        self.training_current_action = training_current_action
        self.training_next_action = training_next_action
        self.testing_pre_data = testing_pre_data
        self.testing_post_data = testing_post_data
        self.testing_current_action = testing_current_action
        self.testing_next_action = testing_next_action

    def one_hot(self,idx, len):
        out = np.zeros((idx.shape[0], len), dtype=np.float)
        for i in range(idx.shape[0]):
            out[i, idx[i]] = 1.0
        return out

    def next_training_batch(self,batch_size):
        x_pre = self.training_pre_data
        x_post = self.training_post_data
        y_current = self.training_current_action
        y_next = self.training_next_action

        x_pre_batch = np.ndarray((batch_size, x_pre.shape[1]))
        x_post_batch = np.ndarray((batch_size, x_post.shape[1]))
        y_current_batch = np.ndarray((batch_size, y_current.shape[1]))
        y_next_batch = np.ndarray((batch_size, y_next.shape[1]))

        for i in range(batch_size):
            idx = (self.batch_idx+i)%x_pre.shape[0]
            x_pre_batch[i] = x_pre[idx]
            x_post_batch[i] = x_post[idx]
            y_current_batch[i] = y_current[idx]
            y_next_batch[i] = y_next[idx]

        # Random augmentation
        aug_pre = np.random.normal(0, 0.1, size=(batch_size, self.training_pre_data.shape[1]))
        aug_post = np.random.normal(0, 0.1, size=(batch_size, self.training_post_data.shape[1]))
        x_pre_batch[:,:self.training_pre_data.shape[1]] = x_pre_batch[:,:self.training_pre_data.shape[1]]+aug_pre
        x_post_batch[:,:self.training_pre_data.shape[1]] = x_post_batch[:,:self.training_pre_data.shape[1]]+aug_post

        self.batch_idx = self.batch_idx + batch_size

        return x_pre_batch, x_post_batch, y_current_batch, y_next_batch

    @property
    def num_examples(self):
        return self.training_pre_data.shape[0]

    @property
    def feature_len(self):
        return self.training_pre_data.shape[1]

    @property
    def num_labels(self):
        return len(self.index_name)


def create_model(n_input, n_classes, train=False):
    def get_scope_variable(scope_name, var, shape, initializer):
        with tf.variable_scope(scope_name) as scope:
            try:
                v = tf.get_variable(var, shape, initializer=initializer)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable(var, shape, initializer=initializer)
        return v

    def create_autoencoder(x):
        # layer_sizes = [64, 16, 32, 128, n_input]
        layer_sizes = [64, 6, n_input]
        enc_index = 1

        # Store layers weight & bias
        weights = [ get_scope_variable('ae', 'weight_0', [n_input, layer_sizes[0]], initializer=tf.random_normal_initializer()) ]
        biases = [ get_scope_variable('ae', 'bias_0', [layer_sizes[0]], initializer=tf.constant_initializer(0.0)) ]

        for i in range(1, len(layer_sizes)):
            weights.append(get_scope_variable('ae', 'weight_{}'.format(i), [layer_sizes[i-1], layer_sizes[i]], initializer=tf.random_normal_initializer()))
            biases.append(get_scope_variable('ae', 'bias_{}'.format(i), [layer_sizes[i]], initializer=tf.constant_initializer(0.0)))


        layer_0 = tf.add(tf.matmul(x, weights[0]), biases[0])
        layer_0 = tf.nn.sigmoid(layer_0)

        enc_layer = layer_0
        last_layer = layer_0

        for i in range(1, len(layer_sizes)-1):
            layer_i = tf.add(tf.matmul(last_layer, weights[i]), biases[i])
            layer_i = tf.nn.sigmoid(layer_i)

            if train:
                layer_i = tf.nn.dropout(layer_i, 0.8)

            # layer_1 = tf.nn.batch_normalization(layer_1, weights['n1_mean'], weights['n1_var'], 0, 0, 1e-3)


            if i == enc_index:
                enc_layer = layer_i
            last_layer = layer_i

        out_layer = tf.matmul(last_layer, weights[-1]) + biases[-1]

        return enc_layer, out_layer

    def create_classifier(x, name):
        with tf.variable_scope(name):
            input_dim = x.get_shape()[1].value

            # weights_0 = tf.get_variable('class_0'.format(name), [input_dim, 64], initializer=tf.random_normal_initializer())
            # biases_0 =  tf.get_variable('bias_0'.format(name), [64], initializer=tf.constant_initializer(0.0))
            # layer_0 = tf.add(tf.matmul(x, weights_0), biases_0)
            # layer_0 = tf.nn.sigmoid(layer_0)
            #
            # weights_1 = tf.get_variable('class_1'.format(name), [64, n_classes], initializer=tf.random_normal_initializer())
            # biases_1 =  tf.get_variable('bias_1'.format(name), [n_classes], initializer=tf.constant_initializer(0.0))
            # layer_1 = tf.add(tf.matmul(layer_0, weights_1), biases_1)

            # return layer_1

            weights_0 = tf.get_variable('class_0'.format(name), [input_dim, n_classes], initializer=tf.random_normal_initializer())
            biases_0 =  tf.get_variable('bias_0'.format(name), [n_classes], initializer=tf.constant_initializer(0.0))
            layer_0 = tf.add(tf.matmul(x, weights_0), biases_0)
            # layer_0 = tf.nn.sigmoid(layer_0)
            #
            # weights_1 = tf.get_variable('class_1'.format(name), [64, n_classes], initializer=tf.random_normal_initializer())
            # biases_1 =  tf.get_variable('bias_1'.format(name), [n_classes], initializer=tf.constant_initializer(0.0))
            # layer_1 = tf.add(tf.matmul(layer_0, weights_1), biases_1)
            return layer_0



    # tf Graph input
    x_pre = tf.placeholder('float', [None, n_input], name='x_pre')
    x_post = tf.placeholder('float', [None, n_input], name='x_post')
    y_current = tf.placeholder('float', [None, n_classes], name='y_current')
    y_next = tf.placeholder('float', [None, n_classes], name='y_next')

    # Construct models
    ae_pre_enc, ae_pre_out = create_autoencoder(x_pre)
    ae_post_enc, ae_post_out = create_autoencoder(x_post)

    ae_enc_combined = tf.concat([ae_pre_enc, ae_post_enc], 1)

    pred_current = create_classifier(ae_enc_combined, 'current')
    pred_next = create_classifier(ae_enc_combined, 'next')


    return x_pre, x_post, y_current, y_next, pred_current, pred_next, ae_pre_enc, ae_post_enc, ae_pre_out, ae_post_out

def train_model():


    dl = DataLoader()

    n_examples = dl.num_examples
    n_input = dl.feature_len
    n_classes = dl.num_labels

    # Parameters
    learning_rate = 0.01
    training_epochs = 5000
    batch_size = 100
    display_step = 50

    x_pre, x_post, y_current, y_next, pred_current, pred_next, _, _, ae_pre_out, ae_post_out = create_model(n_input, n_classes, train=True)

    # Define loss and optimizer
    residual_pre = tf.reduce_mean(tf.squared_difference(x_pre, ae_pre_out))
    residual_post = tf.reduce_mean(tf.squared_difference(x_post, ae_post_out))
    cost_current = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_current, labels=y_current))
    cost_next = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_next, labels=y_next))

    total_loss = 0.01 * (residual_pre + residual_post) + cost_current + cost_next
    # total_loss = cost_next + cost_current

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Calculate accuracy
    correct_pred_current = tf.equal(tf.argmax(pred_current, 1), tf.argmax(y_current, 1))
    correct_pred_next = tf.equal(tf.argmax(pred_next, 1), tf.argmax(y_next, 1))
    accuracy_current = tf.reduce_mean(tf.cast(correct_pred_current, 'float'))
    accuracy_next = tf.reduce_mean(tf.cast(correct_pred_next, 'float'))

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                x_pre_batch, x_post_batch, y_current_batch, y_next_batch = dl.next_training_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                feed = {x_pre: x_pre_batch, x_post: x_post_batch, y_current: y_current_batch, y_next: y_next_batch }
                _, c = sess.run([optimizer, total_loss], feed_dict=feed)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: {:04d} cost: {:.9f}'.format(epoch, avg_cost))
                # print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.training_post_data, y_next: dl.training_next_action})))
                # print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.testing_post_data, y_next: dl.testing_next_action})))
                print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_next: dl.training_next_action})))
                print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_next: dl.testing_next_action})))

                print(' train accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_current: dl.training_current_action})))
                print(' test accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_current: dl.testing_current_action})))

        print("Optimization Finished!")

        if not os.path.exists('./model'):
            os.mkdir('./model')
        saver.save(sess, './model/model.ckpt')

def test_model():

    dl = DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_pre, x_post, y_current, y_next, pred_current, pred_next, _, _, ae_pre_out, ae_post_out = create_model(n_input, n_classes)

    # Calculate accuracy
    correct_pred_current = tf.equal(tf.argmax(pred_current, 1), tf.argmax(y_current, 1))
    correct_pred_next = tf.equal(tf.argmax(pred_next, 1), tf.argmax(y_next, 1))
    accuracy_current = tf.reduce_mean(tf.cast(correct_pred_current, 'float'))
    accuracy_next = tf.reduce_mean(tf.cast(correct_pred_next, 'float'))

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')

        print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_next: dl.training_next_action})))
        print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_next: dl.testing_next_action})))

        print(' train accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_current: dl.training_current_action})))
        print(' test accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_current: dl.testing_current_action})))


'''
Print out the probability tables of the current pre and post-condition observations
'''
def test_sequence():

    dl = DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_pre, x_post, y_current, y_next, pred_current, pred_next, _, _, ae_pre_out, ae_post_out = create_model(n_input, n_classes)

    pred_next_sm = tf.nn.softmax(pred_next)
    pred_current_sm = tf.nn.softmax(pred_current)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')

        for i in range(dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(dl.training_post_data[i,:], axis=0)

            res_current = pred_current_sm.eval({x_pre: x_pre_data, x_post: x_post_data})
            res_next = pred_next_sm.eval({x_pre: x_pre_data, x_post: x_post_data})

            res_current_idx = np.argmax(res_current)
            res_next_idx = np.argmax(res_next)

            print('Prediction current: {} {}, next: {} {}'.format(res_current_idx, dl.index_name[res_current_idx], res_next_idx, dl.index_name[res_next_idx]))
            print(' Probabilities (current next):')
            for j in range(len(dl.index_name)):
                name = dl.index_name[j]
                if len(name) < 7:
                    print(' {}\t\t{:.6f}\t{:.6f}'.format(name, res_current[0,j], res_next[0,j]))
                else:
                    print(' {}\t{:.6f}\t{:.6f}'.format(name, res_current[0,j], res_next[0,j]))
            break

'''
Encode the human measurements into low-dimensional subspace
'''
def encode_human():

    dl = DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_pre, x_post, y_current, y_next, pred_current, pred_next, ae_pre_enc, ae_post_enc, ae_pre_out, ae_post_out = create_model(n_input, n_classes)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')

        for i in range(dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(dl.training_post_data[i,:], axis=0)

            y_enc_pre = ae_pre_enc.eval({x_pre: x_pre_data})
            y_enc_post = ae_post_enc.eval({x_post: x_post_data})

            # Print the 6-dimensional representation
            print(y_enc_pre.tolist())
            print(y_enc_post.tolist())

            break


def parse_args():
    # Parse input arguments
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', default=None, help='train | test | seq | encode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    if args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_model()
    elif args.mode == 'seq':
        test_sequence()
    elif args.mode == 'encode':
        encode_human()
