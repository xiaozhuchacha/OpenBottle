#!/usr/bin/env python

import rospy
from sentence_srv.srv import *
import os
import sys
import re


def produce_sentence(req):
    cmd = sampler_dir + "/SampleSentence.sh " + "\"" + sampler_dir + "\" \"" + grammar_file + "\" \"" + sample_file \
          + "\" " + str(1)
    print "Executing command: %s" % cmd
    try:
        os.system(cmd)

        sentence = parse_sample()

        resp = sentenceResponse()
        resp.sentence = sentence
        resp.success = True
        return resp
    except:
        e = sys.exc_info()[0]
        print "Error: %s" % e
        resp = sentenceResponse()
        resp.sentence = []
        resp.success = False
        return resp

def sentence_server():
    rospy.init_node('sentence_server')

    global sampler_dir
    global grammar_file
    global sample_file

    annotation_mapping_file = rospy.get_param('~annotation_mapping_file')
    sampler_dir = rospy.get_param('~sampler_dir')
    grammar_file = rospy.get_param('~grammar_file')
    sample_file = rospy.get_param('~sample_file')

    print "Sampler dir: %s" % sampler_dir
    print "Sampler script: %s" % sampler_dir + "/SampleSentence.sh"
    print "Grammar file: %s" % grammar_file
    print "Output file: %s" % sample_file

    global annotation_mapping
    annotation_mapping = parse_mapping(annotation_mapping_file)

    s = rospy.Service('get_sentence', sentence, produce_sentence)
    print "Ready to sample sentences"
    rospy.spin()


def parse_sample():
    with open(sample_file) as f:
        content = f.readlines()

    sentence = []
    line = content[-1]
    matches = re.findall('( [0-9]+ |^[0-9]+ )', line)
    matches = [int(match.strip()) + 1 for match in matches]

    for match in matches:
        sentence.append(annotation_mapping[int(match)])

    return sentence


# create the annotation mapping between
def parse_mapping(file):
    with open(file) as f:
        content = f.readlines()

    annotation_mapping = dict()
    for mapping in content:
        pair = mapping.split(',')
        annotation_mapping[int(pair[1])] = pair[0]

    return annotation_mapping

if __name__ == "__main__":
    sentence_server()