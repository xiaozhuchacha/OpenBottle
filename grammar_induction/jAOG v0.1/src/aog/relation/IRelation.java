package aog.relation;

import java.util.ArrayList;

import aog.Pattern;
import aog.TerminalPattern;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * A relation between two patterns
 * 
 * @author Kewei Tu
 * 
 */
public interface IRelation extends Cloneable {

	/**
	 * Return whether the two pattern instances have this relation.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public boolean hasRelation(PatternInstance a, PatternInstance b);

	/**
	 * Return whether the two pattern instances parameterized by the arguments
	 * have this relation.
	 * 
	 * @param typeA
	 * @param parametersA
	 * @param typeB
	 * @param parametersB
	 * @return
	 */
	public boolean hasRelation(Pattern typeA, ArrayList<Double> parametersA,
			Pattern typeB, ArrayList<Double> parametersB);

	/**
	 * Modify this instance to reflect the relation between the two pattern
	 * instances and return it. If no valid relation exists, return null.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public IRelation getRelation(PatternInstance a, PatternInstance b);

	/**
	 * @return a string representation of the relation. Parentheses '(' and ')'
	 *         are not allowed in the string.
	 */
	public String getStringFromRelation();

	public void getRelationFromString(String s);

	public IRelation clone();

	/*
	 * The following methods should have been static, but Interface does not
	 * allow static methods.
	 */
	/**
	 * Return a new parameter of an instance based on its current parameter and
	 * a list of relations (of this relation type) with other instances. Used in
	 * Sampler.
	 * 
	 * @param instance
	 * @param leftRels
	 *            hasRelation(this instance, the other instance)
	 * @param rightRels
	 *            hasRelation(the other instance, this instance)
	 * @return
	 */
	public ArrayList<Double> updateParamFromRelations(PatternInstance instance,
			ArrayList<Pair<IRelation, PatternInstance>> leftRels,
			ArrayList<Pair<IRelation, PatternInstance>> rightRels);

	/**
	 * Return the default parameter of a terminal instance according to the
	 * domain. Used in Sampler.
	 * <p>
	 * TODO This may not be the proper place to put this method. Move to
	 * combiner classes (as getting parameters with no child)?
	 * 
	 * @return
	 */
	public ArrayList<Double> getDefaultParameter(TerminalPattern t);
}
