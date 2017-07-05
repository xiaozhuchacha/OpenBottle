#!/usr/bin/env python

import numpy as np
import scipy.io
import glob
# from keras.layers import Input, Dense, Dropout, Activation
# from keras.models import Model, Sequential, Merge
from keras.utils.np_utils import to_categorical

import tensorflow as tf

class DataLoader:
    def __init__(self):
        self.batch_idx = 0

    def load_data(self, index_name):
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
            next_action_label = np.array(data[0::2, -1]+1)

            current_action_label = np.repeat(index_name.index(mat_file.split('/')[-1][:-12]), pose_pre_data.shape[0])

            print mat_file.split('/')[-1][:-12], pose_pre_data.shape[0]

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
        next_action_label_vec = to_categorical(next_action_label_all, len(index_name))
        current_action_label_all = current_action_label_all[shuffle_index]
        current_action_label_vec = to_categorical(current_action_label_all, len(index_name))

        # split the data
        num_training = int(pose_pre_data_all.shape[0] * 0.8)
        training_data = np.hstack((pose_post_data_all[:num_training, :], force_post_data_all[:num_training, :]))
        training_current_action = current_action_label_vec[:num_training, :]
        training_next_action = next_action_label_vec[:num_training, :]
        testing_data = np.hstack((pose_post_data_all[num_training:, :], force_post_data_all[num_training:, :]))
        testing_current_action = current_action_label_vec[num_training:, :]
        testing_next_action = next_action_label_vec[num_training:, :]

        self.training_data = training_data
        self.training_current_action = training_current_action
        self.training_next_action = training_next_action
        self.testing_data = testing_data
        self.testing_current_action = testing_current_action
        self.testing_next_action = testing_next_action

        # self.training_data_combined = np.concatenate((self.training_data, self.training_current_action), axis=1)
        # self.testing_data_combined = np.concatenate((self.testing_data, self.testing_current_action), axis=1)
        self.training_data_combined = self.training_data
        self.testing_data_combined = self.testing_data


    def next_training_batch(self,batch_size):
        x = self.training_data_combined
        y = self.training_next_action

        x_batch = np.ndarray((batch_size, x.shape[1]))
        y_batch = np.ndarray((batch_size, y.shape[1]))

        for i in range(batch_size):
            idx = (self.batch_idx+i)%x.shape[0]
            x_batch[i] = x[idx]
            y_batch[i] = y[idx]

        aug = np.random.normal(0, 0.1, size=(batch_size, self.training_data.shape[1]))
        x_batch[:,:self.training_data.shape[1]] = x_batch[:,:self.training_data.shape[1]]+aug

        self.batch_idx = self.batch_idx + batch_size
        return x_batch, y_batch


def main():
    index_name = ['end', 'ignore', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                  'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    dl = DataLoader()
    dl.load_data(index_name)

    # training_data, training_current_action, training_next_action, testing_data, testing_current_action, testing_next_action = load_data(index_name)

    # print(training_data[:,0])

    num_examples = dl.training_data.shape[0]

    # Parameters
    learning_rate = 0.001
    training_epochs = 500
    batch_size = 100
    display_step = 100

    # Network Parameters
    n_hidden_1 = 64 # 1st layer number of features
    n_hidden_2 = 64 # 2nd layer number of features
    n_input = dl.training_data_combined.shape[1]
    n_classes = 14 # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        # layer_1 = tf.nn.dropout(layer_1, 0.9)
        # layer_1 = tf.nn.batch_normalization(layer_1, weights['n1_mean'], weights['n1_var'], 0, 0, 1e-3)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        # layer_2 = tf.nn.dropout(layer_2, 0.9)
        # layer_2 = tf.nn.batch_normalization(layer_2, weights['n2_mean'], weights['n2_var'], 0, 0, 1e-3)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)
        # layer_3 = tf.nn.dropout(layer_3, 0.9)
        # layer_3 = tf.nn.batch_normalization(layer_3, weights['n3_mean'], weights['n3_var'], 0, 0, 1e-3)

        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.sigmoid(layer_4)
        # layer_4 = tf.nn.dropout(layer_4, 0.9)
        # layer_4 = tf.nn.batch_normalization(layer_4, weights['n4_mean'], weights['n4_var'], 0, 0, 1e-3)

        layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
        layer_5 = tf.nn.sigmoid(layer_5)
        # layer_5 = tf.nn.dropout(layer_5, 0.9)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, 256])),
        'h2': tf.Variable(tf.random_normal([256, 128])),
        'h3': tf.Variable(tf.random_normal([128, 64])),
        'h4': tf.Variable(tf.random_normal([64, 64])),
        'h5': tf.Variable(tf.random_normal([64, 64])),
        # 'n1_mean': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64])),
        # 'n1_var': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[64])),
        'n2_mean': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128])),
        'n2_var': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[128])),
        'n3_mean': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64])),
        'n3_var': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[64])),
        # 'n4_mean': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
        # 'n4_var': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[32])),
        'out': tf.Variable(tf.random_normal([64, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([256])),
        'b2': tf.Variable(tf.random_normal([128])),
        'b3': tf.Variable(tf.random_normal([64])),
        'b4': tf.Variable(tf.random_normal([64])),
        'b5': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dl.next_training_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
                print("Train accuracy:", accuracy.eval({x: dl.training_data_combined, y: dl.training_next_action}))
                print("Test accuracy:", accuracy.eval({x: dl.testing_data_combined, y: dl.testing_next_action}))

        print("Optimization Finished!")





if __name__ == '__main__':
    main()
