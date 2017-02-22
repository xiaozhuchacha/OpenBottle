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
 * Two instances cooccur if their start/end positions are both the same; they
 * concatenate (follow) if the start of one follows the end of the other.
 * 
 * @author Kewei Tu
 * 
 */
public class CooccurFollowRelation extends AbstractRelation implements
		INegatableRelation {

	protected static enum Type {
		COOCCUR, CONCAT
	}

	public Type type;

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

		if (type == Type.COOCCUR)
			return start1 == start2 && end1 == end2;
		else
			// Type.CONCAT
			return start2 == end1 + 1;
	}

	@Override
	public IRelation getRelation(PatternInstance a, PatternInstance b) {
		if (a.parameters.isEmpty() || b.parameters.isEmpty()) {
			System.out
					.println("[Warning] CooccurFollowRelation: no parameter found in the pattern instances.");
			return null;
		}

		double start1 = a.parameters.get(0);
		double end1 = a.parameters.get(1);
		double start2 = b.parameters.get(0);
		double end2 = b.parameters.get(1);

		if (start1 == start2 && end1 == end2) {
			type = Type.COOCCUR;
			return this;
		} else if (start2 == end1 + 1) {
			type = Type.CONCAT;
			return this;
		} else
			return null;
	}

	@Override
	public String getStringFromRelation() {
		return type.toString();
	}

	@Override
	public void getRelationFromString(String s) {
		type = Enum.valueOf(Type.class, s);
	}

	@Override
	public ArrayList<Double> updateParamFromRelations(PatternInstance instance,
			ArrayList<Pair<IRelation, PatternInstance>> leftRels,
			ArrayList<Pair<IRelation, PatternInstance>> rightRels) {
		if (leftRels.size() + rightRels.size() > 1) {
			System.err
					.println("[Error] updateParamFromRelations(): only one cooccur/follow relation is allowed for an instance");
			System.exit(1);
		}

		if (leftRels.size() == 1) {
			Pair<IRelation, PatternInstance> pair = leftRels.get(0);
			CooccurFollowRelation r = (CooccurFollowRelation) pair.a;
			if (r.type == Type.COOCCUR)
				return new ArrayList<>(pair.b.parameters);
			else { // Type.CONCAT
				int start = instance.parameters.get(0).intValue();
				int end = instance.parameters.get(1).intValue();
				int start2 = pair.b.parameters.get(0).intValue();
				int offset = start2 - 1 - end;
				ArrayList<Double> newParam = new ArrayList<>();
				newParam.add((double) start + offset);
				newParam.add((double) end + offset);
				return newParam;
			}
		} else if (rightRels.size() == 1) {
			Pair<IRelation, PatternInstance> pair = rightRels.get(0);
			CooccurFollowRelation r = (CooccurFollowRelation) pair.a;
			if (r.type == Type.COOCCUR)
				return new ArrayList<>(pair.b.parameters);
			else { // Type.CONCAT
				int start = instance.parameters.get(0).intValue();
				int end = instance.parameters.get(1).intValue();
				int end2 = pair.b.parameters.get(1).intValue();
				int offset = end2 + 1 - start;
				ArrayList<Double> newParam = new ArrayList<>();
				newParam.add((double) start + offset);
				newParam.add((double) end + offset);
				return newParam;
			}
		}

		return new ArrayList<>(instance.parameters);
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
		if (obj instanceof CooccurFollowRelation) {
			CooccurFollowRelation r = (CooccurFollowRelation) obj;
			return type.equals(r.type);
		} else
			return false;
	}

	@Override
	public int hashCode() {
		return type.hashCode();
	}

	@Override
	public INegatableRelation negate() {
		if (type == Type.COOCCUR)
			return this;
		else
			// Type.CONCAT is not negatable
			return null;
	}

}
