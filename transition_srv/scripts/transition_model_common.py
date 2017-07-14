#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy.io
import glob
import os
import csv
import random
import tensorflow as tf
import math

# import sys
# sys.path.append('./tensorflow_hmm')
# import tensorflow_hmm.hmm as hmm


class TransitionModelManager:
    def __init__(self,
                 x_map_input,
                 y_map_output,
                 mapping_weights,
                 x_post,
                 y_current,
                 y_next,
                 pred_next,
                 pred_weights,
                 ae_post_enc,
                 ae_post_out,
                 keep_prob):
        self.x_map_input = x_map_input
        self.y_map_output = y_map_output
        self.mapping_weights = mapping_weights
        self.x_post = x_post
        self.y_current = y_current
        self.y_next = y_next
        self.pred_next = pred_next
        self.pred_weights = pred_weights
        self.ae_post_enc = ae_post_enc
        self.ae_post_out = ae_post_out
        self.keep_prob = keep_prob


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

            # data format: pre followed by post condition row (every other row)
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

    @staticmethod
    def one_hot(idx, len):
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


class RobotDataLoader:
    def __init__(self, dl, x_post, ae_post_enc, keep_prob):

        self.dl = dl

        self.human_enc_dim = 0
        self.robot_dim = 0
        self.feature_len = self.dl.feature_len
        self.num_labels = self.dl.num_labels
        self.index_name = self.dl.index_name

        robot_data_pre = []
        robot_data_post = []
        robot_data_action = []
        robot_data_next_action = []

        files = glob.glob('robot_data/*success*.csv')
        for file in files:
            with open(file, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                last_action = None
                last_pre = None
                for row in reader:
                    data = row[2:9]
                    data[-1] = float(data[-1]) / 255.0
                    action = row[-1]
                    if action != last_action:
                        if last_pre:
                            robot_data_pre.append(last_pre)
                            robot_data_post.append(data)
                            robot_data_action.append(self.index_name.index(last_action))
                            # print(last_action, data[-1])

                        last_pre = data
                        last_action = action

                robot_data_pre.append(last_pre)
                robot_data_post.append(data)
                robot_data_action.append(self.index_name.index(last_action))
                self.robot_dim = len(last_pre)

        # create next actions
        robot_data_next_action = list(robot_data_action)
        robot_data_next_action.pop(0)    # remove first (current action)
        robot_data_next_action.append(0) # append start action

        robot_data_pre = np.array(robot_data_pre, dtype=np.float)
        robot_data_post = np.array(robot_data_post, dtype=np.float)
        robot_data_action = np.array(robot_data_action, dtype=np.uint8)
        robot_data_next_action = np.array(robot_data_next_action, dtype=np.uint8)
        # swap out the next action for pull as 0, the start action (instead of approach)
        robot_data_next_action[np.where(robot_data_next_action==self.index_name.index('approach'))] = 0

        self.robot_action_dict = {}
        self.valid_actions = []
        for i in range(len(self.index_name)):
            robot_data_idxs = robot_data_action==i
            # self.robot_action_dict[i] = (robot_data_pre[robot_data_action==i,:], robot_data_post[robot_data_action==i,:])
            self.robot_action_dict[i] = (None, robot_data_post[robot_data_idxs], robot_data_next_action[robot_data_idxs])
            if robot_data_post[robot_data_action==i].shape[0] > 0:
                self.valid_actions.append(i)


        human_enc_post = []
        human_enc_pre = []
        human_enc_action = []
        human_enc_next_action = []

        human_data_pre = []
        human_data_post = []

        # Encode all the human data
        n_input = self.dl.feature_len
        n_classes = self.dl.num_labels


        for i in range(self.dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(self.dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(self.dl.training_post_data[i,:], axis=0)

            # y_enc_pre = ae_pre_enc.eval({x_pre: x_pre_data})
            y_enc_post = ae_post_enc.eval({x_post: x_post_data, keep_prob: 1.0})

            human_enc_post.append(np.ndarray.flatten(y_enc_post))
            # human_enc_pre.append(y_enc_pre)
            human_enc_action.append(np.argmax(self.dl.training_current_action[i]))
            human_enc_next_action.append(np.argmax(self.dl.training_next_action[i]))
            human_data_pre.append(x_pre_data)
            human_data_post.append(x_post_data)
            self.human_enc_dim = y_enc_post.shape[1]

        human_enc_pre = np.array(human_enc_pre, dtype=np.float)
        human_enc_post = np.array(human_enc_post, dtype=np.float)
        human_enc_action = np.array(human_enc_action, dtype=np.uint8)
        human_enc_next_action = np.array(human_enc_next_action, dtype=np.uint8)
        human_pre = np.array(human_data_pre, dtype=np.float)
        human_post = np.array(human_data_post, dtype=np.float)

        self.human_action_dict = {}
        for i in range(len(self.index_name)):
            action_idxs = human_enc_action==i
            # replace none with pre-condition to use it
            self.human_action_dict[i] = (None, human_enc_post[action_idxs,:], human_enc_next_action[action_idxs])

        self.training_data, self.testing_data = self.generate_training_and_testing_examples(ratio_split=0.8)


    # generates ratio_split examples of training_data and 1-ratio_split
    def generate_training_and_testing_examples(self, ratio_split):
        training_data = {}
        testing_data = {}
        matched_indicies_dict = {}
        for i in range(len(self.valid_actions)):
            action = self.valid_actions[i]
            # match human and robot actions by their next action. first element in tuple is robot, second is human
            # (i.e. training examples will match action and next action for residual
            matched_indicies_dict[action] = self.find_matching_next_actions(action)
            self.print_matched_indicies(matched_indicies_dict)

        for action in matched_indicies_dict:
            matched_len = len(matched_indicies_dict[action])
            training_len = int(math.floor(matched_len * ratio_split))
            # shuffle samples
            np.random.shuffle(matched_indicies_dict[action])
            # split training data according to ratio
            training_pairs = matched_indicies_dict[action][:training_len]
            testing_pairs = matched_indicies_dict[action][training_len:]
            robot_training_idxs = training_pairs[:,0]
            human_training_idxs = training_pairs[:,1]
            robot_testing_idxs = testing_pairs[:,0]
            human_testing_idxs = testing_pairs[:,1]

            ## verify next action matches for all
            assert(np.equal(self.robot_action_dict[action][2][robot_training_idxs], self.human_action_dict[action][2][human_training_idxs]).all())
            assert(np.equal(self.robot_action_dict[action][2][robot_testing_idxs], self.human_action_dict[action][2][human_testing_idxs]).all())

            robot_training_data = self.robot_action_dict[action][1][robot_training_idxs]
            human_training_data = self.human_action_dict[action][1][human_training_idxs]
            training_next_action = self.robot_action_dict[action][2][robot_training_idxs]
            robot_testing_data = self.robot_action_dict[action][1][robot_testing_idxs]
            human_testing_data = self.human_action_dict[action][1][human_testing_idxs]
            testing_next_action = self.robot_action_dict[action][2][robot_testing_idxs]

            # last entry indicates last used training idx
            training_data[action] = np.array([robot_training_data, human_training_data, training_next_action, 0])
            testing_data[action] = np.array([robot_testing_data, human_testing_data, testing_next_action, 0])

        return training_data, testing_data

    def find_matching_next_actions(self, action):
        matched_indicies = []
        robot_next_actions = self.robot_action_dict[action][2]
        human_next_actions = self.human_action_dict[action][2]

        if not np.array_equal(np.unique(robot_next_actions), np.unique(human_next_actions)):
            missing_actions = np.setdiff1d(np.unique(robot_next_actions), np.unique(human_next_actions))
            print("WARNING: no matching robot <-> human training data for action {} with next actions:".format(self.index_name[action]))
            for action_ele in missing_actions:
                print("\t{}".format(self.index_name[action_ele]))

        # for each unique action, collect the corresponding indices
        for ele in np.unique(robot_next_actions):
            robot_matching_idxs = np.where(robot_next_actions == ele)[0]
            human_matching_idxs = np.where(human_next_actions == ele)[0]
            # no human data to match with (warning printed above, in event there is human action not present in robot data)
            if len(human_matching_idxs) == 0:
                continue
            # shuffle indicies before picking the first N (N=length of shorter array)
            np.random.shuffle(robot_matching_idxs)
            np.random.shuffle(human_matching_idxs)

            # pair a human and robot index
            j = 0
            while j < len(robot_matching_idxs) and j < len(human_matching_idxs):
                # first in tuple is robot, second is human
                matched_indicies.append([robot_matching_idxs[j], human_matching_idxs[j]])
                j += 1

        return np.array(matched_indicies)

    def print_matched_indicies(self, matched_indicies_dict):
        for action in matched_indicies_dict:
            matched_indicies = matched_indicies_dict[action]

            for j in range(len(matched_indicies)):
                robot_idx = matched_indicies[j][0]
                human_idx = matched_indicies[j][1]
                next_robot_idx = self.robot_action_dict[action][2][robot_idx]
                next_human_idx = self.human_action_dict[action][2][human_idx]
                print("action: {} robot next action: {} human next action: {}".format(self.index_name[action], self.index_name[next_robot_idx], self.index_name[next_human_idx]))

    def get_random_pair(self):
        human_example = None
        # randomly pick an action

        i = np.random.randint(0, len(self.valid_actions))
        action = self.valid_actions[i]
        while len(self.training_data[action][0]) == 0:
            i = np.random.randint(0, len(self.valid_actions))
            action = self.valid_actions[i]
        self.training_data[action][3] += 1 # count the number of times we have sampled from this action

        idx = np.random.randint(0, len(self.training_data[action][0]))
        robot_example = self.training_data[action][0][idx]
        human_example = self.training_data[action][1][idx]
        next_action = self.training_data[action][2][idx]

        return human_example, robot_example, action, next_action

    def next_training_batch(self, batch_size):
        dat_human = np.zeros((batch_size, self.human_enc_dim), dtype=np.float)
        dat_robot = np.zeros((batch_size, self.robot_dim), dtype=np.float)
        dat_action = np.zeros((batch_size, 1), dtype=np.int)
        dat_next_action = np.zeros((batch_size, 1), dtype=np.int)

        for i in range(batch_size):
            sample_human, sample_robot, action_idx, next_action_idx = self.get_random_pair()
            dat_human[i,:] = sample_human
            dat_robot[i,:] = sample_robot
            dat_action[i,:] = action_idx
            dat_next_action[i,:] = next_action_idx

        return (dat_robot, dat_human, dat_action, dat_next_action)
        # print(dat_human)
        # print(dat_robot)

    def extract_data_as_arrays(self, train=False):
        robot_data = []
        human_data = []
        current_action = []
        next_action = []


        if train:
            data_source = self.training_data
        else:
            data_source = self.testing_data

        for action in self.valid_actions:
            robot_data.extend(data_source[action][0])
            human_data.extend(data_source[action][1])
            current_action.extend(np.full((len(data_source[action][0]),), action))
            next_action.extend(data_source[action][2])

        robot_data = np.array(robot_data)
        human_data = np.array(human_data)
        current_action = np.array(current_action)
        next_action = np.array(next_action)

        return robot_data, human_data, current_action, next_action


def get_scope_variable(scope_name, var, shape, initializer):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var, shape, initializer=initializer)
    return v


def create_mapping_model(x, n_dim1, n_dim2, keep_prob, train=False):
    with tf.variable_scope('mapping'):
        layer_sizes = [256, n_dim2]

        # Store layers weight & bias
        weights = [ get_scope_variable('map', 'weight_0', [n_dim1, layer_sizes[0]], initializer=tf.random_normal_initializer()) ]
        biases = [ get_scope_variable('map', 'bias_0', [layer_sizes[0]], initializer=tf.constant_initializer(0.0)) ]

        for i in range(1, len(layer_sizes)):
            weights.append(get_scope_variable('map', 'weight_{}'.format(i), [layer_sizes[i-1], layer_sizes[i]], initializer=tf.random_normal_initializer()))
            biases.append(get_scope_variable('map', 'bias_{}'.format(i), [layer_sizes[i]],
                                             initializer=tf.constant_initializer(0.0)))

        layer_0 = tf.nn.xw_plus_b(x, weights[0], biases[0])
        layer_0 = tf.nn.relu(layer_0)

        last_layer = layer_0

        # create any remaining layers (e.g. if there are no dropout layers)
        for layer_idx in range(1, len(layer_sizes)-1):
            layer_i = tf.nn.xw_plus_b(last_layer, weights[layer_idx], biases[layer_idx], name="layer_{}".format(layer_idx))
            layer_i = tf.nn.relu(layer_i)

            layer_i = tf.nn.dropout(layer_i, keep_prob=keep_prob)

            # layer_1 = tf.nn.batch_normalization(layer_1, weights['n1_mean'], weights['n1_var'], 0, 0, 1e-3)
            last_layer = layer_i
            layer_idx += 1

        out_layer = tf.nn.xw_plus_b(last_layer, weights[-1], biases[-1], name="out_layer")

        return out_layer, weights


def create_model(n_input, n_classes, train=False):

    enc_size = 8

    def create_autoencoder(x, keep_prob):
        # layer_sizes = [64, 16, 32, 128, n_input]
        layer_sizes = [64, 32, enc_size, 32, 64, n_input]
        enc_index = 2

        # Store layers weight & bias
        weights = [ get_scope_variable('ae', 'weight_0', [n_input, layer_sizes[0]], initializer=tf.random_normal_initializer()) ]
        biases = [ get_scope_variable('ae', 'bias_0', [layer_sizes[0]], initializer=tf.constant_initializer(0.0)) ]

        for i in range(1, len(layer_sizes)):
            weights.append(get_scope_variable('ae', 'weight_{}'.format(i), [layer_sizes[i-1], layer_sizes[i]], initializer=tf.random_normal_initializer()))
            biases.append(get_scope_variable('ae', 'bias_{}'.format(i), [layer_sizes[i]], initializer=tf.constant_initializer(0.0)))

        layer_0 = tf.nn.xw_plus_b(x, weights[0], biases[0], name='layer_0')
        layer_0 = tf.nn.sigmoid(layer_0)
        # layer_0 = tf.nn.relu(layer_0)

        enc_layer = layer_0
        last_layer = layer_0

        for i in range(1, len(layer_sizes)-1):
            layer_i = tf.nn.xw_plus_b(last_layer, weights[i], biases[i], name='layer_{}'.format(i))
            layer_i = tf.nn.sigmoid(layer_i)
            # layer_i = tf.nn.relu(layer_i)

            layer_i = tf.nn.dropout(layer_i, keep_prob=keep_prob)

            # layer_1 = tf.nn.batch_normalization(layer_1, weights['n1_mean'], weights['n1_var'], 0, 0, 1e-3)

            if i == enc_index:
                enc_layer = layer_i

            last_layer = layer_i

        out_layer = tf.nn.xw_plus_b(last_layer, weights[-1], biases[-1], name='out_layer')

        return enc_layer, out_layer

    def create_classifier(x, name):
        with tf.variable_scope(name):
            input_dim = x.get_shape()[1].value

            layer_sizes = [n_classes]

            # Store layers weight & bias
            weights = [ tf.get_variable('weight_0', [input_dim, layer_sizes[0]], initializer=tf.random_normal_initializer()) ]
            biases = [ tf.get_variable('bias_0', [layer_sizes[0]], initializer=tf.constant_initializer(0.0)) ]

            for i in range(1, len(layer_sizes)):
                weights.append(tf.get_variable('weight_{}'.format(i), [layer_sizes[i-1], layer_sizes[i]], initializer=tf.random_normal_initializer()))
                biases.append(tf.get_variable('bias_{}'.format(i), [layer_sizes[i]], initializer=tf.constant_initializer(0.0)))

            # create network
            layer_0 = tf.nn.xw_plus_b(x, weights[0], biases[0], name='layer_0')

            # one layer network, output from input
            if len(layer_sizes) == 1:
                return layer_0, weights

            layer_0 = tf.nn.relu(layer_0)
            last_layer = layer_0

            for i in range(1, len(layer_sizes)-1):
                layer_i = tf.nn.xw_plus_b(last_layer, weights[i], biases[i], name='layer_{}'.format(i))
                layer_i = tf.nn.relu(layer_i)
                last_layer = layer_i

            out_layer = tf.nn.xw_plus_b(last_layer, weights[-1], biases[-1], name='out_layer')

            return out_layer, weights

    with tf.variable_scope('transition'):

        # tf Graph input
        x_pre = tf.placeholder('float', [None, n_input], name='x_pre')
        x_post = tf.placeholder('float', [None, n_input], name='x_post')
        y_current = tf.placeholder('float', [None, n_classes], name='y_current')
        y_next = tf.placeholder('float', [None, n_classes], name='y_next')

        keep_prob = tf.placeholder('float', None, name='keep_prob')

        # Construct models
        # ae_pre_enc, ae_pre_out = create_autoencoder(x_pre)
        ae_post_enc, ae_post_out = create_autoencoder(x_post, keep_prob)

        # ae_enc_combined = tf.concat([ae_pre_enc, ae_post_enc], 1)

        # pred_current = create_classifier(ae_enc_combined, 'current')
        classifier_input = tf.concat([ae_post_enc, y_current], 1)
        pred_next, pred_weights = create_classifier(classifier_input, 'next')

        robot_dim = 7
        x_map_input = tf.placeholder('float', [None, robot_dim], name='x_map_input')
        y_map_output, mapping_weights = create_mapping_model(x_map_input, robot_dim, enc_size, keep_prob, train)

        return TransitionModelManager(x_map_input=x_map_input,
                                      y_map_output=y_map_output,
                                      mapping_weights=mapping_weights,
                                      x_post=x_post,
                                      y_current=y_current,
                                      y_next=y_next,
                                      pred_next=pred_next,
                                      pred_weights=pred_weights,
                                      ae_post_enc=ae_post_enc,
                                      ae_post_out=ae_post_out,
                                      keep_prob=keep_prob)

