#!/usr/bin/env python
import rospy
import rospkg
import subprocess
import os
from aog_update_srv.srv import *


# create the annotation mapping between 
def parse_mapping():

    open_bottle_common_path = rospkg.RosPack().get_path('open_bottle_common')

    file = open_bottle_common_path + '/grammar/annotation_mapping.txt'

    with open(file) as f:
        content = f.readlines()

    annotation_mapping = dict()
    for mapping in content:
        pair = mapping.split(',')
        annotation_mapping[pair[0]] = int(pair[1])

    return annotation_mapping


def update_sentence_count(content):
    for i in range(0, len(content)):
        if content[i] == '\n':
            cur_count = int(content[i+1].strip())
            cur_count += 1
            content[i+1] = str(cur_count) + '\n'
            break

    return content 


def convert_sentence_to_int(sentence, mapping):
    int_sentence = ''
    idx = 0
    for word in sentence:
        action_int = str(mapping[word] - 1)
        int_sentence += action_int + ' [' + str(idx) + ' ' + str(idx) + '] '
        idx += 1

    # print int_sentence
    # return int_sentence + '\n'
    return int_sentence


def update_sentences(sentence):

    annotation_mapping = parse_mapping()

    # print annotation_mapping

    open_bottle_common_path = rospkg.RosPack().get_path('open_bottle_common')

    sentence_file = open_bottle_common_path + "/grammar/sentences.txt"

    with open(sentence_file) as f:
        content = f.readlines()

    with open(sentence_file + '.old', 'w') as f:
        f.writelines(content)

    content = update_sentence_count(content)

    new_sentence = convert_sentence_to_int(sentence, annotation_mapping)

    # append new line to last line if needed
    if content[-1][-1] != '\n':
        content[-1] = content[-1] + ' \n'
        # print content[-1]

    content.append(new_sentence)

    # print content

    with open(sentence_file, 'w') as f:
        f.writelines(content)


def load_aog(file_path):
    lines = [line.rstrip('\n') for line in open(file_path)]

    or_nodes_children = {}
    or_nodes_prob = {}
    for index, line in enumerate(lines):
        # count the number of nodes
        if line.startswith('Terminal#'):
            num_terminal = int(line.lstrip('Terminal#'))
        elif line.startswith('AndNode#'):
            num_and = int(line.lstrip('AndNode#'))
        elif line.startswith('OrNode#'):
            num_or = int(line.lstrip('OrNode#'))
        elif line.startswith('StartSymbol'):
            num_all = int(line.lstrip('StartSymbol')) + 1
            assert num_terminal + num_and + num_or == num_all
        # read nodes
        elif line.startswith('Terminals'):
            terminal_nodes = lines[index+1:index+num_terminal+1]
            terminal_nodes = [int(t.split('\t')[0]) for t in terminal_nodes]
            assert len(terminal_nodes) == num_terminal
        elif line.startswith('AndNodes'):
            and_nodes = lines[index+1:index+num_and+1]
            and_nodes_parent = [int(t.split('\t')[0]) for t in and_nodes]
            and_nodes_left_child = [int(t.split('\t')[1].lstrip('[').split('][')[0].rstrip().split()[0]) for t in and_nodes]
            and_nodes_right_child = [int(t.split('\t')[1].lstrip('[').split('][')[0].rstrip().split()[1]) for t in and_nodes]
            assert len(and_nodes_parent) == num_and
        elif line.startswith('OrNodes'):
            or_nodes = lines[index+1:index+num_or+1]
            or_nodes_parent = [int(t.split('\t')[0]) for t in or_nodes]
            or_nodes_children_raw = [t.split('\t')[1].lstrip('[').split('] [')[0].rstrip().split() for t in or_nodes]
            or_nodes_prob_raw = [t.split('\t')[1].lstrip('[').split('] [')[1].rstrip(' ]').split() for t in or_nodes]
            for i in range(len(or_nodes_parent)):
                or_nodes_children[i] = [int(t) for t in or_nodes_children_raw[i]]
                freq = [float(t) for t in or_nodes_prob_raw[i]]
                or_nodes_prob[i] = [float(t) / sum(freq) for t in or_nodes_prob_raw[i]]
            assert len(or_nodes_parent) == num_or

    root_node = or_nodes_parent[:]
    root_node.extend(and_nodes_parent)
    [root_node.remove(i) for i in and_nodes_left_child]
    [root_node.remove(i) for i in and_nodes_right_child]
    [root_node.remove(i) for i in list(or_nodes_children) if i in root_node]
    assert len(root_node) == 1

    return root_node[0], and_nodes_parent, and_nodes_left_child, and_nodes_right_child, \
           or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_all


def write_parser(root_node, and_nodes_parent, and_nodes_left_child, and_nodes_right_child,
                 or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes, path):
    str_name = ['\'approach\'', '\'move\'', '\'grasp_left\'', '\'grasp_right\'', '\'ungrasp_left\'',
                '\'ungrasp_right\'', '\'twist\'', '\'push\'', '\'neutral\'', '\'pull\'', '\'pinch\'', '\'unpinch\'']
    # padding name to name
    for i in range(len(str_name), num_nodes):
        str_name.append('node_'+str(i))

    with open(path, 'w') as output_file:
        # root
        if root_node in and_nodes_parent:
            root_index = and_nodes_parent.index(root_node)
            output_file.write('root_%d -> %s %s [1.0]\n' % (root_node,
                                                            str_name[and_nodes_left_child[root_index]],
                                                            str_name[and_nodes_right_child[root_index]]))

            del and_nodes_parent[root_index]
            del and_nodes_right_child[root_index]
            del and_nodes_left_child[root_index]
        elif root_node in or_nodes_parent:
            root_index = or_nodes_parent.index(root_node)
            for i, node in enumerate(or_nodes_children[root_index]):
                output_file.write('root_%d ->' % root_node)
                output_file.write(' %s [%s]\n' % (str_name[node], str(or_nodes_prob[root_index][i])))

            del or_nodes_parent[root_index]
            del or_nodes_children[root_index]
            del or_nodes_prob[root_index]
        else:
            print 'no root found in and or parent node'
        # and
        for i in range(len(and_nodes_parent)):
            output_file.write('%s -> %s %s [1.0]\n' % (str_name[and_nodes_parent[i]],
                                                       str_name[and_nodes_left_child[i]],
                                                       str_name[and_nodes_right_child[i]]))
        # or
        for i in range(len(or_nodes_parent)):
            for s, t in zip(or_nodes_children[i], or_nodes_prob[i]):
                output_file.write('%s -> %s [%s]\n' % (str_name[or_nodes_parent[i]], str_name[s], str(t)))


def update_aog(req):
    print 'update aog'
    resp = aog_updateResponse()

    update_flag = req.update
    if not update_flag:
        print 'no need to update aog'
        resp.success = False
        return resp

    update_sentences(req.sentence)

    open_bottle_common_path = rospkg.RosPack().get_path('open_bottle_common')
    command = 'java -cp ' + open_bottle_common_path + '/jAOG/src:' + open_bottle_common_path \
     + '/jAOG/src/args4j-2.33.jar aog.learn.bc.GrammarLearner -combinerType aog.app.sequence.SequenceCombiner -bcRelationType aog.app.sequence.FollowingRelation -bgmRelationType aog.app.sequence.FollowingRelation -contextType aog.app.sequence.SequenceContext -input ' \
     + open_bottle_common_path + '/grammar/sentences.txt -output ' + open_bottle_common_path + '/grammar/grammar.txt'
    # print command 
    subprocess.call(command.split(), stdout=open(os.devnull, 'wb'))

    root_node, and_nodes_parent, and_nodes_left_child, and_nodes_right_child, \
        or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes = load_aog(open_bottle_common_path + '/grammar/grammar.txt')

    write_parser(root_node, and_nodes_parent, and_nodes_left_child, and_nodes_right_child,
                 or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes, open_bottle_common_path + '/grammar/earley_parser_input.txt')

    resp.success = True
    return resp


def update_aog_server():
    rospy.init_node('aog_update')

    s = rospy.Service('update_aog', aog_update, update_aog)
    print 'Ready to update aog'
    rospy.spin()


if __name__ == "__main__":
    update_aog_server()