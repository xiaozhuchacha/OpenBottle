package aog.relation;

import java.util.ArrayList;
import java.util.Collection;

/**
 * This interface defines the relations between an And node and its children.
 * Currently we assume that given the parameters of the children, there exists
 * exactly one assignment of the parameters of the And node such that the
 * relations hold.
 * 
 * @author Kewei Tu
 * 
 */
public interface ICombiner extends Cloneable {

	public ArrayList<Double> getParentFromChildren(
			Collection<ArrayList<Double>> paramOfChildren);

	public String getStringFromCombiner();

	public void getCombinerFromString(String s);

	public ICombiner clone();
}
