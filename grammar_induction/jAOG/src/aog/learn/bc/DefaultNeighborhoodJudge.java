package aog.learn.bc;

import aog.sample.PatternInstance;
import aog.sample.Sample;

public class DefaultNeighborhoodJudge implements NeighborhoodJudge {

	@Override
	public boolean isNeighbor(PatternInstance a, PatternInstance b, Sample s) {
		return true;
	}

}
