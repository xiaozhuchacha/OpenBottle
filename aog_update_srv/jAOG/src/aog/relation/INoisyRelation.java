package aog.relation;

import aog.sample.PatternInstance;

/**
 * A noisy relation defines a distribution over relations.
 * 
 * @author Kewei Tu
 * 
 */
public interface INoisyRelation extends IParametricRelation {

	/**
	 * Check if the two instances have this relation; if they have, then adjust
	 * the parameters of this relation in light of the new instances.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public abstract boolean acceptNewRelation(PatternInstance a,
			PatternInstance b);

	/**
	 * Get the score of the relation between the two instances according to this
	 * relation.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public double getRelationScore(PatternInstance a, PatternInstance b);
	
	/**
	 * Return whether the two relations are approximately equal.
	 * 
	 * @param r
	 * @return
	 */
	public boolean approxEquals(INoisyRelation r);
}
