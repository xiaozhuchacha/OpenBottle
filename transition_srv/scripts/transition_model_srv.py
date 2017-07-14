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
import sys
import transition_model_common as tm

# global variables for callback data, should convert this to a single global class
global prev_precondition    # precondition of completed action
global cur_precondition     # precondition of next action (same as cur_postcondition)
global prev_postcondition    # postcondition of completed action (same as cur_precondition)
global prior_probs 
global force_torque
global cur_action
global gripper_status

cur_precondition = np.random.sample(size=(7,1))
prev_precondition = np.random.sample(size=(7,1))
prev_postcondition = np.random.sample(size=(7,1))
print "cur_precondition: "
print np.transpose(cur_precondition)
print "prev_precondition: "
print np.transpose(prev_precondition)
print "prev_postcondition: "
print np.transpose(prev_postcondition)
cur_action = "start"

# load priors
package_path = rospkg.RosPack().get_path('transition_srv')
prior_probs = scipy.io.loadmat(package_path + '/scripts/prior_counts.mat')['priors']

'''
Print out the probability tables of the current pre and post-condition observations
'''
def trans_prob(req):
    global prev_precondition    # precondition of completed action
    global cur_precondition     # precondition of next action (same as cur_postcondition)
    global prev_postcondition   # postcondition of completed action (same as cur_precondition)
    global prior_probs          # prior probabilities of actions
    global cur_action

    print 'calculating transition probability'
    # print force_called, gripper_called, status_called
    resp = transitionResponse()

    index_name = ['end', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right','twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    if req.reinit:
        cur_precondition = np.random.sample(size=(7,1))
        prev_precondition = np.random.sample(size=(7,1))
        prev_postcondition = np.random.sample(size=(7,1))

    if cur_action == '':
        print('aborting, current action is empty')
        resp.success = False
        return

    n_input = 159
    n_classes = 13
    n_dim1 = 16
    n_dim2 = 7

    x_map_input, y_map_output, x_post, y_current, y_next, pred_next, ae_post_enc, ae_post_out, keep_prob = \
        tm.create_model(n_input, n_classes)
    pred_next_sm = tf.nn.softmax(pred_next)
    # pred_current_sm = tf.nn.softmax(pred_current)

    # Launch the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        package_path = rospkg.RosPack().get_path('transition_srv')
        saver.restore(sess, package_path + '/scripts/models/map/model.ckpt')

        # INSERT ACTUAL ROBOT MEASUREMENTS HERE
        # NOTE: if running from actual robot data, don't forget to divide the gripper
        # state by 255 (last dimension of feature vector)
        # x_robot_pre = copy.deepcopy(np.transpose(prev_precondition))
        x_robot_post = copy.deepcopy(np.transpose(prev_postcondition))

        # y_output_pre = y_map_output.eval({x_map_input: x_robot_pre})
        y_output_post = y_map_output.eval({x_map_input: x_robot_post, keep_prob: 1.0})

        # res_current = pred_current_sm.eval({ae_pre_enc: y_output_pre, ae_post_enc: y_output_post})
        if cur_action == 'start':
            resp.success = False
            return resp
        cur_idx = index_name.index(cur_action)
        cur_one_hot = np.zeros((1, len(index_name)))
        cur_one_hot[0, cur_idx] = 1
        # print("cur one hot: ")
        # print(cur_one_hot)
        # print("%s priors:" % cur_action)
        # print(prior_probs[cur_idx,:])

        res_next = pred_next_sm.eval({ae_post_enc: y_output_post, y_current: cur_one_hot})

        # res_current_idx = np.argmax(res_current)
        res_next_idx = np.argmax(res_next)
        # print('res_next:')
        # print(res_next)

        print('Prediction next: {} {}'.format(res_next_idx, index_name[res_next_idx]))
        print(' Probabilities (next prior):')
        for j in range(len(index_name)):
            name = index_name[j]
            if len(name) < 7:
                print(' {}\t\t{:.6f}\t{:.6f}'.format(name, res_next[0,j], prior_probs[cur_idx,j] ))
            else:
                print(' {}\t{:.6f}\t{:.6f}'.format(name, res_next[0,j], prior_probs[cur_idx,j]))

    output_file = rospkg.RosPack().get_path('open_bottle_common') + '/output/transition_probs.txt'
    with open(output_file, 'a') as f:
        now = rospy.get_rostime()
        # f.write('%i %i %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (now.secs, now.nsecs, res_current[0,0], res_current[0,1],
                 # res_current[0,2], res_current[0,3], res_current[0,4], res_current[0,5], res_current[0,6], res_current[0,7],
                 # res_current[0,8], res_current[0,9], res_current[0,10], res_current[0,11], res_current[0,12]))
        f.write('%i %i %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (now.secs, now.nsecs, res_next[0,0], res_next[0,1],
                 res_next[0,2], res_next[0,3], res_next[0,4], res_next[0,5], res_next[0,6], res_next[0,7],
                 res_next[0,8], res_next[0,9], res_next[0,10], res_next[0,11], res_next[0,12]))
        # print all priors
        f.write('%i %i ' % (now.secs, now.nsecs))
        for i in range(prior_probs.shape[1]):
            f.write('%f ' % prior_probs[cur_idx,i])
        f.write('\n')

    # resp.current = res_current[0,:]
    resp.current = np.zeros(res_next.shape)[0,:]
    resp.next = res_next[0,:]
    resp.prior = prior_probs[cur_idx,:]
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
    
    print "previous action: %s " % cur_action
    cur_action = status.data
    print "current action: %s " % cur_action


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
    
    print "previous precondition: "
    print np.transpose(prev_precondition)
    print "current precondition: "
    print np.transpose(cur_precondition)
    print "previous postcondition: "
    print np.transpose(prev_postcondition)


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

    # global force_called, gripper_called, status_called
    # force_called = gripper_called = status_called = False;
    rospy.spin()


if __name__ == '__main__':
    transition_server()