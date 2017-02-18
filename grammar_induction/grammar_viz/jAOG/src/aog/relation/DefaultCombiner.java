package aog.relation;

import java.util.ArrayList;
import java.util.Collection;

/**
 * The default combiner that simply returns an empty parameter vector for the
 * parent node.
 * 
 * @author Kewei Tu
 * 
 */
public class DefaultCombiner extends AbstractCombiner implements ICombiner {

	@Override
	public ArrayList<Double> getParentFromChildren(
			Collection<ArrayList<Double>> paramOfChildren) {
		return new ArrayList<Double>();
	}

	@Override
	public String getStringFromCombiner() {
		return "";
	}

	@Override
	public void getCombinerFromString(String s) {
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof DefaultCombiner)
			return true;
		else
			return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}
}
