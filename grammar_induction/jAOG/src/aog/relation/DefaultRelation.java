package aog.relation;

import java.util.ArrayList;

import aog.Pattern;
import aog.TerminalPattern;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * Any two pattern instances are considered to have this relation
 * 
 * @author Kewei Tu
 * 
 */
public class DefaultRelation extends AbstractRelation implements
		INegatableRelation {

	@Override
	public boolean hasRelation(PatternInstance a, PatternInstance b) {
		return true;
	}

	@Override
	public boolean hasRelation(Pattern typeA, ArrayList<Double> parametersA,
			Pattern typeB, ArrayList<Double> parametersB) {
		return true;
	}

	@Override
	public String getStringFromRelation() {
		return "";
	}

	@Override
	public void getRelationFromString(String s) {
	}

	@Override
	public IRelation getRelation(PatternInstance a, PatternInstance b) {
		return this;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof DefaultRelation)
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
		return instance.parameters;
	}

	@Override
	public ArrayList<Double> getDefaultParameter(TerminalPattern t) {
		return new ArrayList<>();
	}

	@Override
	public INegatableRelation negate() {
		return this;
	}
}
