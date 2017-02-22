package aog.app.image;

import aog.relation.INoisyRelation;
import aog.sample.PatternInstance;

/**
 * Relative position relation with uniform noise.
 * 
 * @author Kewei Tu
 * 
 */
public class UnifNoisyRelPosRelation extends RelPosRelation implements
		INoisyRelation {

	public double xrange, yrange;

	@Override
	public boolean acceptNewRelation(PatternInstance a, PatternInstance b) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double getRelationScore(PatternInstance a, PatternInstance b) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean approxEquals(INoisyRelation r) {
		// TODO Auto-generated method stub
		return false;
	};
}
