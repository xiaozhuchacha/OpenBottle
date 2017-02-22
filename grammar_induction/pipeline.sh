#!/bin/bash

sentence_dir=$1
annotation_mapping=$2

echo "Sentence directory: ${sentence_dir}"
echo "Annotation mapping: ${annotation_mapping}"

# convert annotations to sentences
python annotation_to_sentence.py "${sentence_dir}" "${annotation_mapping}"

# create AOG grammar from sentences
./SentencesToGrammar.sh "${sentence_dir}/sentences.txt" "${sentence_dir}/grammar.txt"

# create final .dot of grammar
python grammar_viz.py "${sentence_dir}/grammar.txt" "${sentence_dir}/grammar.dot" "${annotation_mapping}"