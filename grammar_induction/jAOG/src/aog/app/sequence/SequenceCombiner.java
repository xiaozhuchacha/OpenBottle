package aog.app.sequence;

import java.util.ArrayList;
import java.util.Collection;

import aog.relation.AbstractCombiner;
import aog.relation.ICombiner;

public class SequenceCombiner extends AbstractCombiner implements ICombiner {

	@Override
	public ArrayList<Double> getParentFromChildren(
			Collection<ArrayList<Double>> paramOfChildren) {
		// we assume that the children constitute a continuous sequence
		double start = Integer.MAX_VALUE, end = Integer.MIN_VALUE;
		for (ArrayList<Double> param : paramOfChildren) {
			double s = param.get(0);
			if (s < start)
				start = s;
			double e = param.get(1);
			if (e > end)
				end = e;
		}

		ArrayList<Double> re = new ArrayList<>(2);
		re.add(start);
		re.add(end);
		return re;
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
		if (obj instanceof SequenceCombiner)
			return true;
		else
			return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}
}
