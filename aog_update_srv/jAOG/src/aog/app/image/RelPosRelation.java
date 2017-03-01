package aog.app.image;

import java.util.ArrayList;
import java.util.Scanner;

import aog.Pattern;
import aog.TerminalPattern;
import aog.relation.AbstractRelation;
import aog.relation.IParametricRelation;
import aog.relation.IRelation;
import aog.sample.PatternInstance;
import aog.util.Pair;
import aog.util.Util;

/**
 * Relative position relation (= b - a).
 * <p>
 * <b>Warning</b>: We use the double data type for the relative positions, which
 * has a precision limit. However, in {@link #equals(Object)} and
 * {@link #hashCode()} we do exact match and do not take the precision issue
 * into account, because of the difficulty in implementing a proper hash code.
 * In the two {@code hasRelation} methods we do consider the precision issue.
 * 
 * @author Kewei Tu
 * 
 */
public class RelPosRelation extends AbstractRelation implements
		IParametricRelation {

	public double dx, dy;

	@Override
	public IRelation getRelation(PatternInstance a, PatternInstance b) {
		if (a.parameters.isEmpty() || b.parameters.isEmpty()) {
			System.out
					.println("[Warning] RelPosRelation: no parameter found in the pattern instances.");
			return null;
		}

		dx = b.parameters.get(0) - a.parameters.get(0);
		dy = b.parameters.get(1) - a.parameters.get(1);
		return this;
	}

	@Override
	public boolean hasRelation(Pattern typeA, ArrayList<Double> parametersA,
			Pattern typeB, ArrayList<Double> parametersB) {
		if (parametersA.isEmpty() || parametersB.isEmpty()) {
			return false;
		}

		return Util.approxEquals(dx, parametersB.get(0) - parametersA.get(0))
				&& Util.approxEquals(dy,
						parametersB.get(1) - parametersA.get(1));
	}

	@Override
	public ArrayList<Double> getBFromA(ArrayList<Double> paramOfA) {
		ArrayList<Double> b = new ArrayList<>();
		b.add(paramOfA.get(0) + dx);
		b.add(paramOfA.get(1) + dy);
		return b;
	}

	@Override
	public ArrayList<Double> getAFromB(ArrayList<Double> paramOfB) {
		ArrayList<Double> a = new ArrayList<>();
		a.add(paramOfB.get(0) - dx);
		a.add(paramOfB.get(1) - dy);
		return a;
	}

	@Override
	public int hashCode() {
		// convert possible -0 to +0
		if (dx == 0)
			dx = 0.0d;
		if (dy == 0)
			dy = 0.0d;

		return Double.valueOf(dx).hashCode() * 31
				+ Double.valueOf(dy).hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof RelPosRelation) {
			RelPosRelation r = (RelPosRelation) obj;
			return dx == r.dx && dy == r.dy;
		} else
			return false;
	}

	@Override
	public IParametricRelation plus(IParametricRelation rel) {
		if (rel instanceof RelPosRelation) {
			RelPosRelation r2 = (RelPosRelation) rel;
			RelPosRelation r = new RelPosRelation();
			r.dx = dx + r2.dx;
			r.dy = dy + r2.dy;
			return r;
		} else
			return null;
	}

	@Override
	public IParametricRelation minus(IParametricRelation rel) {
		if (rel instanceof RelPosRelation) {
			RelPosRelation r2 = (RelPosRelation) rel;
			RelPosRelation r = new RelPosRelation();
			r.dx = dx - r2.dx;
			r.dy = dy - r2.dy;
			return r;
		} else
			return null;
	}

	@Override
	public IParametricRelation negate() {
		RelPosRelation r = new RelPosRelation();
		r.dx = -dx;
		r.dy = -dy;
		return r;
	}

	@Override
	public IParametricRelation divideByScalar(int c) {
		RelPosRelation r = new RelPosRelation();
		r.dx = dx / c;
		r.dy = dy / c;
		return r;
	};

	@Override
	public String toString() {
		return "(" + getStringFromRelation() + ")";
	}

	@Override
	public String getStringFromRelation() {
		return dx + "," + dy;
	}

	@Override
	public void getRelationFromString(String s) {
		Scanner sc = new Scanner(s);
		sc.useDelimiter("[,\\s]+");
		dx = sc.nextDouble();
		dy = sc.nextDouble();
		sc.close();
	}

	@Override
	public ArrayList<Double> updateParamFromRelations(PatternInstance instance,
			ArrayList<Pair<IRelation, PatternInstance>> leftRels,
			ArrayList<Pair<IRelation, PatternInstance>> rightRels) {
		ArrayList<Double> newParam = instance.parameters;
		for (Pair<IRelation, PatternInstance> pair : leftRels) {
			RelPosRelation r = (RelPosRelation) pair.a;
			ArrayList<Double> param = pair.b.parameters;
			if (newParam == instance.parameters)
				newParam = r.getAFromB(param);
			else if (!Util.approxEquals(newParam, r.getAFromB(param))) {
				System.err
						.println("[Error] updateParamFromRelations(): conflicting relations!");
				System.exit(1);
			}
		}
		for (Pair<IRelation, PatternInstance> pair : rightRels) {
			RelPosRelation r = (RelPosRelation) pair.a;
			ArrayList<Double> param = pair.b.parameters;
			if (newParam == instance.parameters)
				newParam = r.getBFromA(param);
			else if (!Util.approxEquals(newParam, r.getBFromA(param))) {
				System.err
						.println("[Error] updateParamFromRelations(): conflicting relations!");
				System.exit(1);
			}
		}
		return newParam;
	}

	@Override
	public ArrayList<Double> getDefaultParameter(TerminalPattern t) {
		ArrayList<Double> param = new ArrayList<>();
		param.add(0d);
		param.add(0d);
		return param;
	}

}
