#!/bin/sh

java -cp ./jAOG/src:./jAOG/src/args4j-2.33.jar aog.learn.bc.GrammarLearner \
	 -combinerType aog.app.sequence.SequenceCombiner \
	 -bcRelationType aog.app.sequence.FollowingRelation \
	 -bgmRelationType aog.app.sequence.FollowingRelation \
	 -contextType aog.app.sequence.SequenceContext \
	 -input $1 -output $2