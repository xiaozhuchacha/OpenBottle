#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy.io
import glob
import os
import csv
import random
import tensorflow as tf
import transition_model_common as tm

# import sys
# sys.path.append('./tensorflow_hmm')
# import tensorflow_hmm.hmm as hmm


def train_model():
    dl = tm.DataLoader()

    n_examples = dl.num_examples
    n_input = dl.feature_len
    n_classes = dl.num_labels

    # Parameters
    learning_rate = 0.01
    training_epochs = 5000
    batch_size = 100
    display_step = 50

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes, train=True)

    # Define loss and optimizer
    # residual_pre = tf.reduce_mean(tf.squared_difference(x_pre, ae_pre_out))
    residual_post = tf.reduce_mean(tf.squared_difference(x_post, ae_post_out))
    # cost_current = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_current, labels=y_current))
    cost_next = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_next, labels=y_next))

    # total_loss = 0.01 * (residual_pre + residual_post) + cost_current + cost_next
    total_loss = 0.01 * (residual_post) + cost_next
    # total_loss = cost_next + cost_current

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Calculate accuracy
    # correct_pred_current = tf.equal(tf.argmax(pred_current, 1), tf.argmax(y_current, 1))
    correct_pred_next = tf.equal(tf.argmax(pred_next, 1), tf.argmax(y_next, 1))
    # accuracy_current = tf.reduce_mean(tf.cast(correct_pred_current, 'float'))
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
                # feed = {x_pre: x_pre_batch, x_post: x_post_batch, y_current: y_current_batch, y_next: y_next_batch }
                feed = {x_post: x_post_batch, y_current: y_current_batch, y_next: y_next_batch }
                _, c = sess.run([optimizer, total_loss], feed_dict=feed)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: {:04d} cost: {:.9f}'.format(epoch, avg_cost))
                # print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.training_post_data, y_next: dl.training_next_action})))
                # print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.testing_post_data, y_next: dl.testing_next_action})))
                print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.training_post_data,
                                                                                  y_current: dl.training_current_action,
                                                                                  y_next: dl.training_next_action})))
                print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.testing_post_data,
                                                                                 y_current: dl.testing_current_action,
                                                                                 y_next: dl.testing_next_action})))

                # print(' train accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_current: dl.training_current_action})))
                # print(' test accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_current: dl.testing_current_action})))

                test_action_accuracy(accuracy_next, x_post, y_current, y_next, dl, training=False)

        print("Optimization Finished!")

        if not os.path.exists('./models/transition'):
            os.mkdir('./models/transition')
        saver.save(sess, './models/transition/model.ckpt')


def train_mapping():

    dl = tm.DataLoader()
    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/transition/model.ckpt')

        rdl = tm.RobotDataLoader(dl, x_post, ae_post_enc)

        n_dim1 = rdl.human_enc_dim
        n_dim2 = rdl.robot_dim

        # tf Graph input
        # x = tf.placeholder('float', [None, n_dim2], name='x_robot_enc')
        y_gt = tf.placeholder('float', [None, n_dim1], name='y_human_gt')
        # y = create_mapping_model(x, n_dim2, n_dim1, train=True)
        x = x_map_input
        y = y_map_output

        # Parameters
        learning_rate = 0.001
        training_epochs = 5000
        batch_size = 100
        display_step = 50
        total_batch = 20

        # Calculate accuracy
        # correct_pred_current = tf.equal(tf.argmax(pred_current, 1), tf.argmax(y_current, 1))
        correct_pred_next = tf.equal(tf.argmax(pred_next, 1), tf.argmax(y_next, 1))
        # accuracy_current = tf.reduce_mean(tf.cast(correct_pred_current, 'float'))
        accuracy_next = tf.reduce_mean(tf.cast(correct_pred_next, 'float'))

        # Define loss and optimizer
        residual = tf.reduce_mean(tf.squared_difference(y, y_gt))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(residual)

        new_vars = []
        for var in tf.global_variables():
            if 'mapping' in var.name or 'beta' in var.name:
                new_vars.append(var)

        # Initializing the variables
        # init = tf.global_variables_initializer()
        init = tf.initialize_variables(new_vars)

        # Launch the graph
        sess.run(init)

        num_training = training_epochs * total_batch * batch_size
        # robot data projected to human subspace
        mapped_robot_data = np.zeros((num_training, n_dim1), dtype=np.float)
        action_idx_data = np.full((num_training, len(dl.index_name)), -2, dtype=np.int)
        next_action_idx_data = np.full((num_training, len(dl.index_name)), -2, dtype=np.int)
        data_idx = 0
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                x_batch, y_batch, action_idx_batch, next_action_idx_batch = rdl.next_training_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                feed = {x: x_batch, y_gt: y_batch, keep_prob: 0.5}
                _, c = sess.run([optimizer, residual], feed_dict=feed)
                # Compute average loss
                avg_cost += c / total_batch

                # collect data to feed to accuracy eval
                mapped_robot_data[data_idx:data_idx+batch_size,:] = y_map_output.eval({x_map_input: x_batch,
                                                                                       keep_prob: 1.0})
                action_idx_data[data_idx:data_idx+batch_size,:] = dl.one_hot(action_idx_batch, len(dl.index_name))
                next_action_idx_data[data_idx:data_idx+batch_size,:] = dl.one_hot(next_action_idx_batch, len(dl.index_name))
                data_idx += batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: {:04d} cost: {:.9f}'.format(epoch, avg_cost))

                print(' accuracy (next): {:.9f}'.format(accuracy_next.eval({ae_post_enc: mapped_robot_data[0:data_idx],
                                                                            y_current: action_idx_data[0:data_idx],
                                                                            y_next: next_action_idx_data[0:data_idx]})))
                test_action_accuracy_map(accuracy_next, ae_post_enc, y_current, y_next,
                                         mapped_robot_data[0:data_idx], action_idx_data[0:data_idx],
                                         next_action_idx_data[0:data_idx], dl)

        print("Optimization Finished!")

        if not os.path.exists('./models/map'):
            os.mkdir('./models/map')

        saver.save(sess, './models/map/model.ckpt')


'''
Map from robot state to human (encoded) state
'''
def run_mapping():

    dl = tm.DataLoader()
    n_input = dl.feature_len
    n_classes = dl.num_labels
    n_dim1 = 6
    n_dim2 = 7

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes, train=True)
    pred_next_sm = tf.nn.softmax(pred_next)
    # pred_current_sm = tf.nn.softmax(pred_current)

    # tf Graph input
    # x = tf.placeholder('float', [None, n_dim2], name='x_robot_enc')
    # y = create_mapping_model(x, n_dim2, n_dim1, train=False)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/map/model.ckpt')

        rdl = tm.RobotDataLoader(dl, x_post, ae_post_enc)


        for i in range(10):
            y_human, x_robot, action_idx, next_action_idx = rdl.get_random_pair()

            # y_output_pre = y_map_output.eval({x_map_input: np.expand_dims(x_robot[0], axis=0)})
            y_action = dl.one_hot(np.full((1,), action_idx), len(dl.index_name))
            y_output_post = y_map_output.eval({x_map_input: np.reshape(x_robot, (1,7)),
                                               keep_prob: 1.0})

            # res_current = pred_current_sm.eval({ae_pre_enc: y_output_pre, ae_post_enc: y_output_post})
            res_next = pred_next_sm.eval({ae_post_enc: y_output_post,
                                          y_current: y_action,
                                          keep_prob: 1.0})

            # res_current_idx = np.argmax(res_current)
            res_next_idx = np.argmax(res_next)

            print('Prediction next: {} {}, true {} {}'.format(res_next_idx, dl.index_name[res_next_idx],
                                                              next_action_idx, dl.index_name[next_action_idx]))
            print(' Probabilities (next):')
            for j in range(len(dl.index_name)):
                name = dl.index_name[j]
                tab_str = get_tab_str(name)
                print(' {}{}{:.6f}'.format(name, tab_str, res_next[0,j]))


def run_demo():

    index_name = ['end', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                  'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    n_input = 159
    n_classes = 13
    n_dim1 = 6
    n_dim2 = 7

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes, train=True)
    pred_next_sm = tf.nn.softmax(pred_next)
    # pred_current_sm = tf.nn.softmax(pred_current)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/map/model.ckpt')

        # INSERT ACTUAL ROBOT MEASUREMENTS HERE
        # NOTE: if running from actual robot data, don't forget to divide the gripper
        # state by 255 (last dimension of feature vector)
        x_robot_pre = np.random.normal(size=(1,7))
        x_robot_post = np.random.normal(size=(1,7))
        action = np.full((1,), random.randint(1,13))
        y_action = tm.DataLoader.one_hot(action, 13)

        # y_output_pre = y_map_output.eval({x_map_input: x_robot_pre})
        y_output_post = y_map_output.eval({x_map_input: x_robot_post,
                                           y_current: y_action,
                                           keep_prob: 1.0})

        # res_current = pred_current_sm.eval({ae_pre_enc: y_output_pre, ae_post_enc: y_output_post})
        res_next = pred_next_sm.eval({ae_post_enc: y_output_post,
                                      y_current: y_action,
                                      keep_prob: 1.0})

        # res_current_idx = np.argmax(res_current)
        res_next_idx = np.argmax(res_next)

        print('Prediction next: {} {}'.format(res_next_idx, index_name[res_next_idx]))
        print(' Probabilities (next):')
        for j in range(len(index_name)):
            name = index_name[j]
            tab_str = get_tab_str(name)
            print(' {}{}{:.6f}'.format(name, tab_str, res_next[0,j]))

'''
Tests the accuracy of a single action's encoding/decoding
'''
def test_action_accuracy(accuracy_next, x_post, y_current, y_next, dl=tm.DataLoader(),
                         training=False):
    if training:
        # pre_data = dl.training_pre_data
        post_data = dl.training_post_data
        current_action = dl.training_current_action
        next_action = dl.training_next_action
        type_str = 'training'
    else:
        # pre_data = dl.testing_pre_data
        post_data = dl.testing_post_data
        current_action = dl.testing_current_action
        next_action = dl.testing_next_action
        type_str = 'testing'

    for action_idx in range(1, len(dl.index_name)):
        # find matching indicies for this action
        index_arr = np.full((1, 1), action_idx)
        action_one_hot = dl.one_hot(index_arr, len(dl.index_name))
        action_indices = np.where((current_action == action_one_hot).all(axis=1))[0]

        tab_str = get_tab_str(dl.index_name[action_idx])
        print(' {}:{} {} accuracy (next): {:.9f}'.format(dl.index_name[action_idx],
                                                         tab_str,
                                                         type_str,
                                                         accuracy_next.eval({x_post: post_data[action_indices,:],
                                                                             y_current: current_action[action_indices,:],
                                                                             y_next: next_action[action_indices,:]})))


'''
Tests the accuracy of a single action's encoding/decoding during mapping
'''
def test_action_accuracy_map(accuracy_next, ae_post_enc, y_current, y_next, mapped_robot_data,
                             action_idx_data, next_action_idx_data, dl=tm.DataLoader()):

    for action_idx in range(1, len(dl.index_name)):
        # find matching indicies for this action
        index_arr = np.full((1, 1), action_idx)
        action_one_hot = dl.one_hot(index_arr, len(dl.index_name))
        action_indices = np.where((action_idx_data == action_one_hot).all(axis=1))[0]

        tab_str = get_tab_str(dl.index_name[action_idx])
        print(' {}:{} accuracy (next): {:.9f}'.format(dl.index_name[action_idx],
                                                      tab_str,
                                                      accuracy_next.eval({ae_post_enc: mapped_robot_data[action_indices,:],
                                                                          y_current: action_idx_data[action_indices,:],
                                                                          y_next: next_action_idx_data[action_indices,:]})))


def test_model():

    dl = tm.DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes)

    # Calculate accuracy
    # correct_pred_current = tf.equal(tf.argmax(pred_current, 1), tf.argmax(y_current, 1))
    correct_pred_next = tf.equal(tf.argmax(pred_next, 1), tf.argmax(y_next, 1))
    # accuracy_current = tf.reduce_mean(tf.cast(correct_pred_current, 'float'))
    accuracy_next = tf.reduce_mean(tf.cast(correct_pred_next, 'float'))

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/map/model.ckpt')

        print(' train accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.training_post_data,
                                                                          y_current: dl.training_current_action,
                                                                          y_next: dl.training_next_action})))
        print(' test accuracy (next): {:.9f}'.format(accuracy_next.eval({x_post: dl.testing_post_data,
                                                                         y_current: dl.testing_current_action,
                                                                         y_next: dl.testing_next_action})))

        # print(' train accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.training_pre_data, x_post: dl.training_post_data, y_current: dl.training_current_action})))
        # print(' test accuracy (current): {:.9f}'.format(accuracy_current.eval({x_pre: dl.testing_pre_data, x_post: dl.testing_post_data, y_current: dl.testing_current_action})))


'''
Print out the probability tables of the current pre and post-condition observations
'''
def test_sequence():

    dl = tm.DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    # x_map_input, y_map_output, x_post, y_current, y_next, pred_current, pred_next, ae_pre_enc, ae_post_enc, ae_pre_out, ae_post_out = tm.create_model(n_input, n_classes)
    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes)

    pred_next_sm = tf.nn.softmax(pred_next)
    # pred_current_sm = tf.nn.softmax(pred_current)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/transition/model.ckpt')

        for i in range(dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(dl.training_post_data[i,:], axis=0)
            y_action = np.reshape(dl.training_current_action[i,:], (1, len(dl.index_name)))

            # res_current = pred_current_sm.eval({x_pre: x_pre_data, x_post: x_post_data})
            # res_next = pred_next_sm.eval({x_pre: x_pre_data, x_post: x_post_data})
            res_next = pred_next_sm.eval({x_post: x_post_data,
                                          y_current: y_action})

            # res_current_idx = np.argmax(res_current)
            res_next_idx = np.argmax(res_next)

            print('Prediction next: {} {}'.format(res_next_idx, dl.index_name[res_next_idx]))
            print(' Probabilities (next):')
            for j in range(len(dl.index_name)):
                name = dl.index_name[j]
                tab_str = get_tab_str(name)
                print(' {}:{}{:.6f}'.format(name, tab_str, res_next[0,j]))
            break


'''
Encode the human measurements into low-dimensional subspace
'''
def encode_human():

    dl = tm.DataLoader()

    n_input = dl.feature_len
    n_classes = dl.num_labels

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/transition/model.ckpt')

        for i in range(dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(dl.training_post_data[i,:], axis=0)

            # y_enc_pre = ae_pre_enc.eval({x_pre: x_pre_data})
            y_enc_post = ae_post_enc.eval({x_post: x_post_data})

            # Print the 6-dimensional representation
            # print(y_enc_pre.tolist())
            print(y_enc_post.tolist())

            break


def get_tab_str(action_name):
    if len(action_name) < 7:
        tab_str = '\t\t\t'
    elif len(action_name) >= 7 and len(action_name) < 10:
        tab_str = '\t\t'
    else:
        tab_str = '\t'

    return tab_str



def parse_args():
    # Parse input arguments
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', default=None, help='train | trainmap | runmap | test | seq | encode')

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
    elif args.mode == 'trainmap':
        train_mapping()
    elif args.mode == 'runmap':
        run_mapping()
    elif args.mode == 'demo':
        run_demo()
