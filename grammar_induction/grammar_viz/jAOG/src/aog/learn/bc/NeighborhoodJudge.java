package aog.learn.bc;

import aog.sample.PatternInstance;
import aog.sample.Sample;

public interface NeighborhoodJudge {
	/**
	 * The judgment should be symmetrical.
	 * 
	 * @param a
	 * @param b
	 * @param s
	 * @return
	 */
	public boolean isNeighbor(PatternInstance a, PatternInstance b, Sample s);
}
