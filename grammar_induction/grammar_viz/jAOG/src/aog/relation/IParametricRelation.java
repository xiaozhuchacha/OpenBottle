package aog.relation;

import java.util.ArrayList;

/**
 * A relation with parameters. Typically it is set to (B - A).
 * 
 * @author Kewei Tu
 * 
 */
public interface IParametricRelation extends INegatableRelation {

	/**
	 * Get the parameters of instance b if a and b have this relation.
	 * 
	 * @param paramOfA
	 * @return
	 */
	public ArrayList<Double> getBFromA(ArrayList<Double> paramOfA);

	/**
	 * Get the parameters of instance a if a and b have this relation.
	 * 
	 * @param b
	 * @return
	 */
	public ArrayList<Double> getAFromB(ArrayList<Double> paramOfB);

	public IParametricRelation plus(IParametricRelation rel);

	public IParametricRelation minus(IParametricRelation rel);

	@Override
	public IParametricRelation negate();

	public IParametricRelation divideByScalar(int c);
}
