jAOG
v0.1
Dec 2013

This package contains a Java implementation of And-Or grammars (AOG), including the basic data structures, a parser, a sampler and a learner. Further development is ongoing with more advanced learning algorithms. A more detailed documentation will be provided in the future.

==========
The AOG learning approach included in the current package is described in the following paper:
Kewei Tu, Maria Pavlovskaia and Song-Chun Zhu, "Unsupervised Structure Learning of Stochastic And-Or Grammars". In NIPS 2013.

The main class of the learning approach is aog.learn.bc.GrammarLearner. 
The most important options are:
 -input VAL                     : The input corpus file name
 -output VAL                    : The output grammar file name
 -bcRelationType VAL            : The full class name of the relation used in
                                  the And-nodes
 -bgmRelationType VAL           : The full class name of the relation between
								  pattern instances
 -combinerType VAL              : The full class name of the combiner in the
                                  grammar
 -contextType VAL               : The full class name of the context type
 -contextRange N                : The context range
 -alpha N                       : A parameter of the grammar prior probability
 -dl1 N                         : A parameter of the grammar prior probability
 -dl2 N                         : A parameter of the grammar prior probability

 
=== The format of a grammar file ===
"Terminal#" followed by the number of terminals
"AndNode#" followed by the number of And nodes
"OrNode#" followed by the number of Or nodes
"StartSymbol" followed by the ID of the Or node representing the start symbol
Relation type
"Terminals"
A list of terminal patterns, one per line, in the format of "ID String"
"AndNodes"
A list of And nodes, one per line, in the format of "ID [list of child_IDs][list of relations between children][combiner]" where the list elements are separated by space
"OrNodes"
A list of Or nodes, one per line, in the format of "ID [list of child_IDs] [list of probabilities]" where the list elements are separated by space


=== The format of a corpus file ===
Terminal pattern list: the first line is the number of terminal patterns; then the list of all the individual terminal patterns in the corpus, one per line.
A blank line.
Sample list: the first line is the number of samples; then the list of all the samples, one per line. Each sample is a sequence of terminal patterns (represented by their 0-based line numbers in the term list) and their parameters (in a pair of brackets, separated by space), e.g., "1 [0.1 1.2] 5 [1.5 3.3]".
