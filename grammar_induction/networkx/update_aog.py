import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import read_dot
from networkx.drawing.nx_pydot import graphviz_layout


def load_aog(file_path):
    lines = [line.rstrip('\n') for line in open(file_path)]
    G = nx.Graph()

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

            [G.add_node(t, type='terminal') for t in terminal_nodes]
        elif line.startswith('AndNodes'):
            and_nodes = lines[index+1:index+num_and+1]
            and_nodes_parent = [int(t.split('\t')[0]) for t in and_nodes]
            and_nodes_left_child = [int(t.split('\t')[1].lstrip('[').split('][')[0].rstrip().split()[0]) for t in and_nodes]
            and_nodes_right_child = [int(t.split('\t')[1].lstrip('[').split('][')[0].rstrip().split()[1]) for t in and_nodes]
            assert len(and_nodes_parent) == num_and

            [G.add_node(t, type='and') for t in and_nodes_parent]
        elif line.startswith('OrNodes'):
            or_nodes = lines[index+1:index+num_or+1]
            or_nodes_parent = [int(t.split('\t')[0]) for t in or_nodes]
            or_nodes_children = [t.split('\t')[1].lstrip('[').split('] [')[0].rstrip().split() for t in or_nodes]
            or_nodes_prob = [t.split('\t')[1].lstrip('[').split('] [')[1].rstrip(' ]').split() for t in or_nodes]
            assert len(or_nodes_parent) == num_or

            [G.add_node(t, type='or') for t in or_nodes_parent]

    and_edge_list = []
    for index, and_node in enumerate(and_nodes_parent):
        G.add_edge(and_node, and_nodes_left_child[index], prob=1.0)
        and_edge_list.append([and_node, and_nodes_left_child[index]])
        G.add_edge(and_node, and_nodes_right_child[index], prob=1.0)
        and_edge_list.append([and_node, and_nodes_right_child[index]])
    or_edge_list = []
    for index, or_node in enumerate(or_nodes_parent):
        [G.add_edge(or_node, int(t), prob=float(s)/100.0) for t, s in zip(or_nodes_children[index], or_nodes_prob[index])]
        [or_edge_list.append([or_node, int(t)]) for t, s in zip(or_nodes_children[index], or_nodes_prob[index])]

    # draw
    pos = graphviz_layout(G)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=terminal_nodes,
                           node_color='r',
                           node_size=500)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=and_nodes_parent,
                           node_color='y',
                           node_size=500)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=or_nodes_parent,
                           node_color='g',
                           node_size=500)
    nx.draw_networkx_edges(G, pos,
                           edgelist=and_edge_list,
                           width=8, alpha=0.8, edge_color='y')
    nx.draw_networkx_edges(G, pos,
                           edgelist=or_edge_list,
                           width=8, alpha=0.8, or_edge_list='g')
    labels = {}
    for i in range(num_all):
        labels[i] = str(i)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.show()

    print G.number_of_nodes()
    print G.number_of_edges()


def main():
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_grammar')
    parser.add_argument('-o', '--output_grammar')
    args = parser.parse_args()

    input_grammar = args.input_grammar
    assert (os.path.exists(input_grammar))
    assert (os.path.isfile(input_grammar))

    # G = nx.Graph(read_dot(input_grammar))

    load_aog(input_grammar)


if __name__ == '__main__':
    main()

