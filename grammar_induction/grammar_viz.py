# dot grammar.dot -Tpng > grammar.png

import sys


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
        terminals = []
        num_terminals = int(lines[0].split(' ')[-1])
        num_and = int(lines[1].split(' ')[-1])
        num_or = int(lines[2].split(' ')[-1])
        start_symbol = int(lines[3].split(' ')[-1])

        with open(viz_file, 'wb+') as f_out:
            f_out.write('digraph G {\n')
            for i in range(num_terminals):
                f_out.write(str(sorted_annotations[i]) + ' [shape=box, fillcolor=lightgrey, style=filled, ranksep=0.5, '
                                                   'nodesep=0.5]\n')

            idx_and = 7 + num_terminals + 1
            idx_or = idx_and + num_and + 1

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

            for i in range(idx_and, idx_and + num_and):
                line_and = lines[i]
                root_and = line_and.split('\t')[0]
                children_and = line_and.split(' ]')[0].split('\t')[1][1:].split(' ')
                f_out.write(root_and + ' [shape=doublecircle, fillcolor=palegreen3, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
                for child_and in children_and:
                    print(root_and + ' -> ' + child_and)
                    fixed_child_and = child_and
                    if root_and == '31':
                        print("hey")
                    if child_and in single_ors:
                        fixed_child_and = single_ors[child_and]
                    if int(child_and) in range(0, len(sorted_annotations)):
                        fixed_child_and = sorted_annotations[int(child_and)]
                    print(root_and + ' -> ' + fixed_child_and)
                    f_out.write(root_and + ' -> ' + fixed_child_and + ' [penwidth=3, weight=3]\n')

            for i in range(idx_or, idx_or + num_or):
                line_or = lines[i]
                root_or = line_or.split('\t')[0]
                if root_or in single_ors:
                    continue
                f_out.write(root_or + ' [shape=circle, fillcolor=palegreen1, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
                children_or = line_or.split(' ]')[0].split('\t')[1][1:].split(' ')
                weights_or = line_or.split(' ]')[1][2:].split(' ')
                sum_weights = 0
                for i in range(len(children_or)):
                    sum_weights += float(weights_or[i])
                for i in range(len(children_or)):
                    normalized_weight = float(weights_or[i])/sum_weights
                    penwidth = 1. + 2.*normalized_weight
                    fixed_child_or = children_or[i]
                    if children_or[i] in single_ors:
                        fixed_child_or = single_ors[children_or[i]]
                    if int(children_or[i]) in range(0, len(sorted_annotations)):
                        fixed_child_or = sorted_annotations[int(children_or[i])]
                    f_out.write(root_or + ' -> ' + fixed_child_or + ' [label = "' + "{0:.2f}".format(
                        normalized_weight) + '", penwidth=' + str(penwidth) + ', weight=3 ]\n')
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

