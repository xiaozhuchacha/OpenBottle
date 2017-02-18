package aog.app.sequence;

import java.util.ArrayList;

import aog.Pattern;
import aog.TerminalPattern;
import aog.relation.AbstractRelation;
import aog.relation.IRelation;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * This relation holds if b is right after a in the sequence.
 * 
 * @author Kewei Tu
 * 
 */
public class FollowingRelation extends AbstractRelation {

	@Override
	public IRelation getRelation(PatternInstance a, PatternInstance b) {
		if (a.parameters.isEmpty() || b.parameters.isEmpty()) {
			System.out
					.println("[Warning] FollowingRelation: no parameter found in the pattern instances.");
			return null;
		}

		// int start1 = a.parameters.get(0).intValue();
		int end1 = a.parameters.get(1).intValue();
		int start2 = b.parameters.get(0).intValue();
		// int end2 = b.parameters.get(1).intValue();

		if (start2 == end1 + 1)
			return this;
		else
			return null;
	}

	@Override
	public boolean hasRelation(Pattern typeA, ArrayList<Double> parametersA,
			Pattern typeB, ArrayList<Double> parametersB) {
		if (parametersA.isEmpty() || parametersB.isEmpty()) {
			return false;
		}

		// double start1 = parametersA.get(0);
		double end1 = parametersA.get(1);
		double start2 = parametersB.get(0);
		// double end2 = parametersB.get(1);

		return start2 == end1 + 1;
	}

	@Override
	public String getStringFromRelation() {
		return "";
	}

	@Override
	public void getRelationFromString(String s) {
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof FollowingRelation)
			return true;
		else
			return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}

	@Override
	public ArrayList<Double> updateParamFromRelations(PatternInstance instance,
			ArrayList<Pair<IRelation, PatternInstance>> leftRels,
			ArrayList<Pair<IRelation, PatternInstance>> rightRels) {
		if (leftRels.size() + rightRels.size() > 1) {
			System.err
					.println("[Error] updateParamFromRelations(): only one following relation is allowed for an instance");
			System.exit(1);
		}

		int start = instance.parameters.get(0).intValue();
		int end = instance.parameters.get(1).intValue();

		int offset = 0;
		if (leftRels.size() == 1) {
			int start2 = leftRels.get(0).b.parameters.get(0).intValue();
			offset = start2 - 1 - end;
		} else if (rightRels.size() == 1) {
			int end2 = rightRels.get(0).b.parameters.get(1).intValue();
			offset = end2 + 1 - start;
		}

		ArrayList<Double> newParam = new ArrayList<>();
		newParam.add((double) start + offset);
		newParam.add((double) end + offset);
		return newParam;
	}

	@Override
	public ArrayList<Double> getDefaultParameter(TerminalPattern t) {
		ArrayList<Double> param = new ArrayList<>();
		param.add(1d);
		param.add(1d);
		return param;
	}
}
