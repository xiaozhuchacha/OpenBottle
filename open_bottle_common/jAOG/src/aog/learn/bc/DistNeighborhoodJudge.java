package aog.learn.bc;

import aog.sample.PatternInstance;
import aog.sample.Sample;

/**
 * Distance based NeighborhoodJudge.
 * 
 * @author Kewei Tu
 * 
 */
public abstract class DistNeighborhoodJudge implements NeighborhoodJudge {

	public double maxDist = Double.POSITIVE_INFINITY;

	/**
	 * We assume that the first two parameters of the input PatternInstance are
	 * the x/y coordinates.
	 */
	@Override
	public boolean isNeighbor(PatternInstance a, PatternInstance b, Sample s) {
		if (a.parameters.isEmpty() || b.parameters.isEmpty()) {
			System.out
					.println("[Warning] DistNeighborhoodJudge: no parameter found in the pattern instances.");
			return true;
		}

		return getDistance(a, b, s) <= maxDist;
	}

	protected abstract double getDistance(PatternInstance a, PatternInstance b,
			Sample s);

	public void setMaxDist(double maxDist) {
		this.maxDist = maxDist;
	}

}
