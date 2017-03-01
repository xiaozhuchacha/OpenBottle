import argparse
import os


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
                 or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes):
    str_name = ['\'approach\'', '\'move\'', '\'grasp_left\'', '\'grasp_right\'', '\'ungrasp_left\'',
                '\'ungrasp_right\'', '\'twist\'', '\'push\'', '\'neutral\'', '\'pull\'', '\'pinch\'', '\'unpinch\'']
    # padding name to name
    for i in range(len(str_name), num_nodes):
        str_name.append('node_'+str(i))

    with open('parser_input.txt', 'w') as output_file:
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
            output_file.write('root_%d ->' % root_node)
            for i in or_nodes_children[root_index]:
                output_file.write(' %s %s\n' % (str_name[i], str(or_nodes_prob[root_index])))

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


def main():
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_grammar')
    args = parser.parse_args()

    input_grammar = args.input_grammar
    assert (os.path.exists(input_grammar))
    assert (os.path.isfile(input_grammar))

    root_node, and_nodes_parent, and_nodes_left_child, and_nodes_right_child, \
        or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes = load_aog(input_grammar)

    write_parser(root_node, and_nodes_parent, and_nodes_left_child, and_nodes_right_child,
                 or_nodes_parent, or_nodes_children, or_nodes_prob, terminal_nodes, num_nodes)


if __name__ == '__main__':
    main()

