#!/usr/bin/env python
import rospy
import rospkg
from transition_srv.srv import *
from transition_srv.msg import *
from std_msgs.msg import String
from baxter_core_msgs.msg import EndpointState


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy.io
import glob
import os
import csv
import random
import tensorflow as tf
import copy

# global variables for callback data, should convert this to a single global class
global prev_precondition    # precondition of completed action
global cur_precondition     # precondition of next action (same as cur_postcondition)
global prev_postcondition    # postcondition of completed action (same as cur_precondition)
global force_torque
global cur_action
global gripper_status

cur_precondition = np.zeros(shape=(7,1))
prev_precondition = np.zeros(shape=(7,1))
prev_postcondition = np.zeros(shape=(7,1))
cur_action = "start"

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

        package_path = rospkg.RosPack().get_path('transition_srv')
        mat_files = glob.glob(package_path + '/scripts/data/*.mat')
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


class RobotDataLoader:
    def __init__(self, dl, x_pre, x_post, ae_pre_enc, ae_post_enc):

        self.dl = dl

        self.human_enc_dim = 0
        self.robot_dim = 0
        self.feature_len = self.dl.feature_len
        self.num_labels = self.dl.num_labels
        self.index_name = self.dl.index_name


        data_pre = []
        data_post = []
        data_action = []

        package_path = rospkg.RosPack().get_path('transition_srv')
        files = glob.glob(package_path + '/robot_data/*success.csv')
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
                            data_pre.append(last_pre)
                            data_post.append(data)
                            data_action.append(self.index_name.index(last_action))
                            # print(last_action, data[-1])

                        last_pre = data
                        last_action = action

                data_pre.append(last_pre)
                data_post.append(data)
                data_action.append(self.index_name.index(last_action))
                self.robot_dim = len(last_pre)

        data_pre = np.array(data_pre, dtype=np.float)
        data_post = np.array(data_post, dtype=np.float)
        data_action = np.array(data_action, dtype=np.uint8)

        self.action_dict = {}
        self.valid_actions = []
        for i in range(len(self.index_name)):
            self.action_dict[i] = (data_pre[data_action==i,:], data_post[data_action==i,:])
            if data_pre[data_action==i].shape[0] > 0:
                self.valid_actions.append(i)


        self.human_enc_post = []
        self.human_enc_pre = []
        self.human_enc_action = []

        # Encode all the human data
        n_input = self.dl.feature_len
        n_classes = self.dl.num_labels

        for i in range(self.dl.training_pre_data.shape[0]):
            x_pre_data = np.expand_dims(self.dl.training_pre_data[i,:], axis=0)
            x_post_data = np.expand_dims(self.dl.training_post_data[i,:], axis=0)

            y_enc_pre = ae_pre_enc.eval({x_pre: x_pre_data})
            y_enc_post = ae_post_enc.eval({x_post: x_post_data})

            self.human_enc_post.append(y_enc_post)
            self.human_enc_pre.append(y_enc_pre)
            self.human_enc_action.append(np.argmax(self.dl.training_current_action[i]))
            self.human_enc_dim = y_enc_pre.shape[1]


    def get_random_pair(self):
        human_example = None
        action_idx = -1
        while True:
            idx = random.randint(0, len(self.human_enc_pre)-1)
            pre_data = self.human_enc_pre[idx]
            post_data = self.human_enc_post[idx]
            action = self.human_enc_action[idx]

            if action in self.valid_actions:
                success = True
                action_idx = action
                human_example = (pre_data, post_data)
                break

        robot_idx = random.randint(0, self.action_dict[action_idx][0].shape[0]-1)
        robot_example = (self.action_dict[action_idx][0][robot_idx], self.action_dict[action_idx][1][robot_idx])
        return human_example, robot_example, action_idx


    def next_training_batch(self, batch_size):
        dat_human = np.zeros((batch_size, self.human_enc_dim), dtype=np.float)
        dat_robot = np.zeros((batch_size, self.robot_dim), dtype=np.float)

        for i in range(batch_size):
            idx = random.randint(0,1) #select pre or post randomly
            sample_human, sample_robot, _ = self.get_random_pair()
            dat_human[i,:] = sample_human[idx]
            dat_robot[i,:] = sample_robot[idx]

        return (dat_robot, dat_human)
        # print(dat_human)
        # print(dat_robot)


def get_scope_variable(scope_name, var, shape, initializer):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var, shape, initializer=initializer)
    return v

def create_mapping_model(x, n_dim1, n_dim2, train=False):
    with tf.variable_scope('mapping'):
        layer_sizes = [12, 12, 12, 12, n_dim2]

        # Store layers weight & bias
        weights = [ get_scope_variable('map', 'weight_0', [n_dim1, layer_sizes[0]], initializer=tf.random_normal_initializer()) ]
        biases = [ get_scope_variable('map', 'bias_0', [layer_sizes[0]], initializer=tf.constant_initializer(0.0)) ]

        for i in range(1, len(layer_sizes)):
            weights.append(get_scope_variable('map', 'weight_{}'.format(i), [layer_sizes[i-1], layer_sizes[i]], initializer=tf.random_normal_initializer()))
            biases.append(get_scope_variable('map', 'bias_{}'.format(i), [layer_sizes[i]], initializer=tf.constant_initializer(0.0)))

        layer_0 = tf.add(tf.matmul(x, weights[0]), biases[0])
        layer_0 = tf.nn.relu(layer_0)

        last_layer = layer_0
        for i in range(1, len(layer_sizes)-1):
            layer_i = tf.add(tf.matmul(last_layer, weights[i]), biases[i])
            layer_i = tf.nn.relu(layer_i)

            # if train:
            #     layer_i = tf.nn.dropout(layer_i, 0.8)

            # layer_1 = tf.nn.batch_normalization(layer_1, weights['n1_mean'], weights['n1_var'], 0, 0, 1e-3)
            last_layer = layer_i

        out_layer = tf.matmul(last_layer, weights[-1]) + biases[-1]

        return out_layer


def create_model(n_input, n_classes, train=False):

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

            weights_0 = tf.get_variable('class_0', [input_dim, n_classes], initializer=tf.random_normal_initializer())
            biases_0 =  tf.get_variable('bias_0', [n_classes], initializer=tf.constant_initializer(0.0))
            layer_0 = tf.add(tf.matmul(x, weights_0), biases_0)

            # layer_0 = tf.nn.sigmoid(layer_0)
            #
            # weights_1 = tf.get_variable('class_1'.format(name), [64, n_classes], initializer=tf.random_normal_initializer())
            # biases_1 =  tf.get_variable('bias_1'.format(name), [n_classes], initializer=tf.constant_initializer(0.0))
            # layer_1 = tf.add(tf.matmul(layer_0, weights_1), biases_1)
            return layer_0

    with tf.variable_scope('transition'):

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

        x_map_input = tf.placeholder('float', [None, 7], name='x_map_input')
        y_map_output = create_mapping_model(x_map_input, 7, 6, train)

        return x_map_input, y_map_output, x_pre, x_post, y_current, y_next, pred_current, pred_next, ae_pre_enc, ae_post_enc, ae_pre_out, ae_post_out


'''
Print out the probability tables of the current pre and post-condition observations
'''
def trans_prob(req):
    print 'calculating transition probability'
    print force_called, gripper_called, status_called
    resp = transitionResponse()

    index_name = ['end', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                  'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    n_input = 159
    n_classes = 13
    n_dim1 = 6
    n_dim2 = 7

    x_map_input, y_map_output, x_pre, x_post, y_current, y_next, pred_current, pred_next, ae_pre_enc, ae_post_enc, ae_pre_out, ae_post_out = create_model(n_input, n_classes)
    pred_next_sm = tf.nn.softmax(pred_next)
    pred_current_sm = tf.nn.softmax(pred_current)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        package_path = rospkg.RosPack().get_path('transition_srv')
        saver.restore(sess, package_path + '/scripts/models/map/model.ckpt')

        # INSERT ACTUAL ROBOT MEASUREMENTS HERE
        # NOTE: if running from actual robot data, don't forget to divide the gripper
        # state by 255 (last dimension of feature vector)
        x_robot_pre = copy.deepcopy(np.transpose(prev_precondition))
        x_robot_post = copy.deepcopy(np.transpose(prev_postcondition))

        y_output_pre = y_map_output.eval({x_map_input: x_robot_pre})
        y_output_post = y_map_output.eval({x_map_input: x_robot_post})

        res_current = pred_current_sm.eval({ae_pre_enc: y_output_pre, ae_post_enc: y_output_post})
        res_next = pred_next_sm.eval({ae_pre_enc: y_output_pre, ae_post_enc: y_output_post})

        res_current_idx = np.argmax(res_current)
        res_next_idx = np.argmax(res_next)

        print('Prediction current: {} {}, next: {} {}'.format(res_current_idx, index_name[res_current_idx], res_next_idx, index_name[res_next_idx]))
        print(' Probabilities (current next):')
        for j in range(len(index_name)):
            name = index_name[j]
            if len(name) < 7:
                print(' {}\t\t{:.6f}\t{:.6f}'.format(name, res_current[0,j], res_next[0,j]))
            else:
                print(' {}\t{:.6f}\t{:.6f}'.format(name, res_current[0,j], res_next[0,j]))

    resp = transition_srvResponse()
    resp.current = 
    resp.success = True
    return resp


def gripperCallback_left(gripper_input):
    global gripper_status

    gripper_status = gripper_input.gPOA


def execStatusCallback(status):
    global prev_precondition    # precondition of completed action
    global cur_precondition     # precondition of next action (same as cur_postcondition)
    global prev_postcondition    # postcondition of completed action (same as cur_precondition)
    global force_torque
    global cur_action
    global gripper_status
    
    print "prev_action: %s " % cur_action
    cur_action = status.data
    print "cur_action: %s " % cur_action


    # update pre and post conditions
    prev_precondition = copy.deepcopy(cur_precondition) # previous precondition is the last action's precondition (we are about to update cur_precondition)

    cur_precondition[0] = copy.deepcopy(force_torque['force_x'])
    cur_precondition[1] = copy.deepcopy(force_torque['force_y'])
    cur_precondition[2] = copy.deepcopy(force_torque['force_z'])
    cur_precondition[3] = copy.deepcopy(force_torque['torque_x'])
    cur_precondition[4] = copy.deepcopy(force_torque['torque_y'])
    cur_precondition[5] = copy.deepcopy(force_torque['torque_z'])
    cur_precondition[6] = float(gripper_status) / 255     # normalize between 0 and 1
    
    # update the previous post condition
    prev_postcondition = copy.deepcopy(cur_precondition)
    
    print "prev_precondition: "
    print prev_precondition
    print "cur_precondition: "
    print cur_precondition
    print "prev_postcondition: "
    print prev_postcondition


def forceCallback_left(endpoint_state):
    global force_torque

    force_torque = dict()
    force_torque['force_x'] = endpoint_state.wrench.force.x;
    force_torque['force_y'] = endpoint_state.wrench.force.y;
    force_torque['force_z'] = endpoint_state.wrench.force.z;

    force_torque['torque_x'] = endpoint_state.wrench.torque.x;
    force_torque['torque_y'] = endpoint_state.wrench.torque.y;
    force_torque['torque_z'] = endpoint_state.wrench.torque.z;


def transition_server():
    rospy.init_node('transition_server')

    s = rospy.Service('get_transition', transition, trans_prob)
    print 'transition server is ready'

    rospy.Subscriber("/SModelRobotInput", SModel_robot_input, gripperCallback_left)
    rospy.Subscriber("/execution_status", String, execStatusCallback)
    rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState, forceCallback_left)

    global force_called, gripper_called, status_called
    force_called = gripper_called = status_called = False;
    rospy.spin()


if __name__ == '__main__':
    transition_server()