import time
import nltk


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
            if p_edge.start() + p_edge.dot() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
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


def main():
    grammar = read_induced_grammar('../grammars/parser_input.txt')
    sentence = 'approach'
    tokens = sentence.split()
    print predict_next_symbols(grammar, tokens)


if __name__ == '__main__':
    main()
