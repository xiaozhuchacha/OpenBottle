#!/usr/bin/env python
import rospy
import rospkg
import nltk
from action_earley_srv.srv import *


def read_induced_grammar(path):
    with open(path) as f:
        rules = [rule.strip() for rule in f.readlines()]
        grammar = nltk.PCFG.fromstring(rules)
        return grammar


def predict_next_symbols(grammar, tokens):
    def get_production_prob(selected_edge):
        # Find the corresponding production rule of the edge, and return its probability
        for production in grammar.productions(lhs=selected_edge.lhs()):
            if production.rhs() == selected_edge.rhs():
                # print selected_edge, production.prob()
                return production.prob()

    def find_parent(selected_edge):
        # Find the parent edges that lead to the selected edge
        p_edges = list()
        for p_edge in e_chart.edges():
            if p_edge.end() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
                p_edges.append(p_edge)
        return p_edges

    def get_edge_prob(selected_edge):
        # Compute the probability of the edge by recursion
        prob = get_production_prob(selected_edge)
        if selected_edge.start() != 0:
            parent_prob = 0
            for parent_edge in find_parent(selected_edge):
                parent_prob += get_edge_prob(parent_edge)
            prob *= parent_prob
        return prob

    symbols = list()
    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    e_chart = earley_parser.chart_parse(tokens)
    end_edges = list()

    for edge in e_chart.edges():
        # print edge
        if edge.end() == len(tokens):
            # Only add terminal nodes
            if isinstance(edge.nextsym(), unicode):
                symbols.append(edge.nextsym())
                end_edges.append(edge)

    probs = list()
    for end_edge in end_edges:
        probs.append(get_edge_prob(end_edge))

    # Eliminate duplicate
    symbols_no_duplicate = list()
    probs_no_duplicate = list()
    for s, p in zip(symbols, probs):
        if s not in symbols_no_duplicate:
            symbols_no_duplicate.append(s)
            probs_no_duplicate.append(p)
        else:
            probs_no_duplicate[symbols_no_duplicate.index(s)] += p

    return zip(symbols_no_duplicate, probs_no_duplicate)


def probability(req):
    print 'calculate probability'

    grammar = read_induced_grammar(rospkg.RosPack().get_path('open_bottle_common') + '/grammar/earley_parser_input.txt')
    sentence = req.action
    print 'input sentence is', sentence

    tokens = sentence.split()
    prediction = predict_next_symbols(grammar, tokens)

    action_label = []
    action_prob = []
    action_prob_raw = []
    for i in range(len(prediction)):
    	action_label.append(str(prediction[i][0]))
    	action_prob_raw.append(float(prediction[i][1]))
    prob_sum = sum(action_prob_raw)
    for i in range(len(prediction)):
    	action_prob.append(action_prob_raw[i] / prob_sum)

    resp = action_earleyResponse()
    resp.action_seq = action_label
    resp.action_prob = action_prob
    resp.success = True
    return resp
    

def actioin_earley_server():
    rospy.init_node('action_earley_server')

    s = rospy.Service('get_earley_action', action_earley, probability)
    print 'earley service is ready'
    rospy.spin()


if __name__ == "__main__":
    actioin_earley_server()