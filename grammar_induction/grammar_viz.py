# dot grammar.dot -Tpng > grammar.png

import sys


class Node:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children

    def __str__(self):
        if self.children is None:
            children_str = "None"
        else:
            children_str = "[" + ','.join(self.children) + "]"
        return "{name: " + self.name + ", children: " + children_str + "}"

    def __repr__(self):
        return str(self)

class OrNode(Node):
    def __init__(self, name, children=None, weights=None):
        super(OrNode, self).__init__(name, children)
        self.weights = weights


class AndNode(Node):
    def __init__(self, name, children=None):
        super(AndNode, self).__init__(name, children)


class AOG:
    def __init__(self, root_node=None):
        self.root_node = root_node
        self.node_dict = {}
        if root_node is not None:
            self.node_dict[self.root_node.name] = self.root_node

    def traverse_bfs(self):
        node_order = [self.root_node]
        self._travserse_bfs(self.root_node, node_order)
        return node_order

    def _travserse_bfs(self, node, node_order):
        nodes_to_traverse = []
        if node.children is None:
            return
        for child in node.children:
            # only add to order if this is the first time seeing this child
            if self.node_dict[child] not in node_order:
                node_order.append(self.node_dict[child])
                nodes_to_traverse.append(self.node_dict[child])

        for child in nodes_to_traverse:
            self._travserse_bfs(child, node_order)


def main():

    grammar_file = sys.argv[1]
    viz_file = sys.argv[2]
    annotation_mapping_file = sys.argv[3]

    annotation_mapping = parse_mapping(annotation_mapping_file)

    sorted_annotations = len(annotation_mapping)*[None]

    # build sorted reverse of str->int action dictionary
    for identifier in annotation_mapping:
        sorted_annotations[annotation_mapping[identifier]-1] = identifier

    with open(grammar_file, 'r+') as f_in:
        raw_data = f_in.read()
        lines = raw_data.split('\n')
        num_terminals = int(lines[0].split(' ')[-1])
        num_and = int(lines[1].split(' ')[-1])
        num_or = int(lines[2].split(' ')[-1])
        start_symbol = int(lines[3].split(' ')[-1])

        node_dict = {}
        aog = AOG()

        idx_and = 7 + num_terminals + 1
        idx_or = idx_and + num_and + 1

        for i in range(num_terminals):
            aog.node_dict[str(sorted_annotations[i])] = OrNode(str(sorted_annotations[i]))
            node_dict[str(sorted_annotations[i])] = [str(sorted_annotations[i]) + ' [shape=box, fillcolor=grey90, style="rounded,filled", ranksep=0.5, nodesep=0.5]\n']

        # find single-children OR nodes
        single_ors = {}
        for i in range(idx_or, idx_or + num_or):
            line_or = lines[i]
            root_or = line_or.split('\t')[0]
            children_or = line_or.split(' ]')[0].split('\t')[1][1:].split(' ')
            if len(children_or) == 1:
                # use a label from the sorted annotations
                if int(children_or[0]) in range(0, len(sorted_annotations)):
                    single_ors[root_or] = sorted_annotations[int(children_or[0])]
                else:
                    single_ors[root_or] = children_or[0]

                # node_dict[root_or] = single_ors[root_or]
                # aog.node_dict[root_or] = OrNode(root_or, [single_ors[root_or]])

        # process ands
        for i in range(idx_and, idx_and + num_and):
            line_and = lines[i]
            root_and = line_and.split('\t')[0]
            children_and = line_and.split(' ]')[0].split('\t')[1][1:].split(' ')
            processed_children_and = []
            node_dict[root_and] = [root_and + ' [shape=circle, fillcolor="#7ce57b", style=filled, color=black, ranksep=0.5, nodesep=0.5]\n']
            for child_and in children_and:
                print(root_and + ' -> ' + child_and)
                fixed_child_and = child_and
                # convert to single or or terminal
                if child_and in single_ors:
                    fixed_child_and = single_ors[child_and]
                if int(child_and) in range(0, len(sorted_annotations)):
                    fixed_child_and = sorted_annotations[int(child_and)]

                processed_children_and.append(fixed_child_and)
                print(root_and + ' -> ' + fixed_child_and)
                node_dict[root_and].append(root_and + ' -> ' + fixed_child_and + ' [penwidth=3, weight=3]\n')

            aog.node_dict[root_and] = AndNode(root_and, processed_children_and)

        # process or's
        for i in range(idx_or, idx_or + num_or):
            line_or = lines[i]
            root_or = line_or.split('\t')[0]
            if root_or in single_ors:
                continue
            node_dict[root_or] = [root_or + ' [shape=circle, fillcolor="#8cb7ff", style=filled, color=black, ranksep=0.5, nodesep=0.5]\n']
            children_or = line_or.split(' ]')[0].split('\t')[1][1:].split(' ')
            weights_or = line_or.split(' ]')[1][2:].split(' ')
            processed_children_or = []
            sum_weights = 0
            for i in range(len(children_or)):
                sum_weights += float(weights_or[i])
            for i in range(len(children_or)):
                normalized_weight = float(weights_or[i])/sum_weights
                penwidth = 1. + 2.*normalized_weight
                fixed_child_or = children_or[i]
                # convert to single or or terminal
                if children_or[i] in single_ors:
                    fixed_child_or = single_ors[children_or[i]]
                if int(children_or[i]) in range(0, len(sorted_annotations)):
                    fixed_child_or = sorted_annotations[int(children_or[i])]

                processed_children_or.append(fixed_child_or)
                node_dict[root_or].append(root_or + ' -> ' + fixed_child_or + ' [label = "' + "{0:.2f}".format(normalized_weight) + '", penwidth=' + str(penwidth) + ', weight=3 ]\n')

            normalized_weights_or = [float(x)/sum_weights for x in weights_or]
            aog.node_dict[root_or] = OrNode(root_or, processed_children_or, normalized_weights_or)

        aog.root_node = aog.node_dict[str(start_symbol)]
        # determine order of nodes to write using bfs
        node_order = aog.traverse_bfs()
        node_order_str = [x.name for x in node_order]

        with open(viz_file, 'w+') as f_out:
            f_out.write('digraph G {\n')
            f_out.write('graph[fontname = "DejaVu Sans"];\n')
            f_out.write('node[fontname = "DejaVu Sans"];\n')
            f_out.write('edge[fontname = "DejaVu Sans"];\n')
            for node_str in node_order_str:
                for node_line in node_dict[node_str]:
                    f_out.write(node_line)
            # add in legend
            # legend_str = \
            #     'subgraph legend { \n' \
            #     '  label="Legend"\n' \
            #     '  or_legend[label="", shape=circle, fillcolor="#8cb7ff", style=filled, color=black, ranksep=0.5, nodesep=0.5]\n' \
            #     '  or_text[shape=plaintext, style=solid, label="Or node\\r", width=1.5] // Add fixed width so all nodes line up\n' \
            #     '  and_legend[label="", shape=circle, fillcolor="#7ce57b", style=filled, color=black, ranksep=0.5, nodesep=0.5]\n' \
            #     '  and_text[shape=plaintext, style=solid, label="And node\\r", width=1.5] // Add fixed width so all nodes line up\n' \
            #     '  terminal_legend[label="", shape=box, fillcolor=grey90, style="rounded,filled", ranksep=0.5, nodesep=0.5]\n' \
            #     '  terminal_text[shape=plaintext, style=solid, label="Terminal node\\r", width=1.5] // Add fixed width so all nodes line up\n' \
            #     '  { rank=source;or_legend and_legend terminal_legend }\n' \
            #     '}\n'
            # f_out.write(legend_str)
            f_out.write('}')

# create the annotation mapping between
def parse_mapping(file):
    with open(file) as f:
        content = f.readlines()

    annotation_mapping = dict()
    for mapping in content:
        pair = mapping.split(',')
        annotation_mapping[pair[0]] = int(pair[1])

    return annotation_mapping


if __name__ == "__main__":
    main()
