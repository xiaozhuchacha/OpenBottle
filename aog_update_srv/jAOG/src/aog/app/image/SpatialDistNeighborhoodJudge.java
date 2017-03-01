package aog.app.image;

import aog.learn.bc.DistNeighborhoodJudge;
import aog.sample.PatternInstance;
import aog.sample.Sample;

public class SpatialDistNeighborhoodJudge extends DistNeighborhoodJudge {

	@Override
	protected double getDistance(PatternInstance a, PatternInstance b, Sample s) {
		double dx = a.parameters.get(0) - b.parameters.get(0);
		double dy = a.parameters.get(1) - b.parameters.get(1);
		return Math.sqrt(dx * dx + dy * dy);
	}
}
