#!/bin/bash

sampler_dir=$1
input_grammar=$2
output_file=$3
num_samples=$4

cd $sampler_dir

java -cp ./jAOG/out/production/jAOG/:./jAOG/out/production/jAOG/args4j-2.33.jar aog.sample.Sampler "${input_grammar}" "${output_file}" $num_samples

cd -