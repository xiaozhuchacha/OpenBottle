package aog.app.vectorseq;

import java.util.ArrayList;

import aog.Pattern;
import aog.TerminalPattern;
import aog.relation.AbstractRelation;
import aog.relation.INegatableRelation;
import aog.relation.IRelation;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * Two instances cooccur if their start/end positions are both the same.
 * 
 * @author Kewei Tu
 * 
 */
public class CooccurRelation extends AbstractRelation implements
		INegatableRelation {

	@Override
	public boolean hasRelation(Pattern typeA, ArrayList<Double> parametersA,
			Pattern typeB, ArrayList<Double> parametersB) {
		if (parametersA.isEmpty() || parametersB.isEmpty()) {
			return false;
		}

		double start1 = parametersA.get(0);
		double end1 = parametersA.get(1);
		double start2 = parametersB.get(0);
		double end2 = parametersB.get(1);

		return start1 == start2 && end1 == end2;
	}

	@Override
	public IRelation getRelation(PatternInstance a, PatternInstance b) {
		if (a.parameters.isEmpty() || b.parameters.isEmpty()) {
			System.out
					.println("[Warning] CooccurRelation: no parameter found in the pattern instances.");
			return null;
		}

		double start1 = a.parameters.get(0);
		double end1 = a.parameters.get(1);
		double start2 = b.parameters.get(0);
		double end2 = b.parameters.get(1);

		if (start1 == start2 && end1 == end2)
			return this;
		else
			return null;
	}

	@Override
	public String getStringFromRelation() {
		return "";
	}

	@Override
	public void getRelationFromString(String s) {
	}

	@Override
	public ArrayList<Double> updateParamFromRelations(PatternInstance instance,
			ArrayList<Pair<IRelation, PatternInstance>> leftRels,
			ArrayList<Pair<IRelation, PatternInstance>> rightRels) {
		if (leftRels.size() + rightRels.size() > 1) {
			System.err
					.println("[Error] updateParamFromRelations(): only one cooccur relation is allowed for an instance");
			System.exit(1);
		}

		ArrayList<Double> p = instance.parameters;
		if (leftRels.size() == 1) {
			p = leftRels.get(0).b.parameters;
		} else if (rightRels.size() == 1) {
			p = rightRels.get(0).b.parameters;
		}
		return new ArrayList<>(p);
	}

	@Override
	public ArrayList<Double> getDefaultParameter(TerminalPattern t) {
		ArrayList<Double> param = new ArrayList<>();
		param.add(0d);
		param.add(0d);
		return param;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof CooccurRelation)
			return true;
		else
			return false;
	}

	@Override
	public int hashCode() {
		return 0;
	}

	@Override
	public INegatableRelation negate() {
		return this;
	}

}
