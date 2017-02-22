package aog.app.image;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import aog.Pattern;
import aog.learn.bc.CompositionalContext;
import aog.learn.bc.Context;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * Distance based compositional context, defined by a circle around the center.
 * 
 * @author Kewei Tu
 * 
 */
public class CompositionalDistContext extends CompositionalContext {

	protected Set<Pair<Pattern, ArrayList<Double>>> signature0;

	public CompositionalDistContext() {
		super();
	}

	public CompositionalDistContext(PatternInstance pi) {
		super(pi);
		signature = signature0;
	}

	public CompositionalDistContext(Context c) {
		if (c instanceof CompositionalDistContext) {
			CompositionalDistContext cdc = (CompositionalDistContext) c;
			signature0 = new HashSet<>(cdc.signature0);
			signature = signature0;
		} else {
			System.err.println("[CompositionalDistContext] type error.");
			System.exit(1);
		}
	}

	@Override
	protected boolean contains(PatternInstance query) {
		if (centerParameters.isEmpty() || query.parameters.isEmpty()) {
			System.out
					.println("[Warning] DistContext: no parameter found in the pattern instances.");
			return false;
		}

		double dx = centerParameters.get(0) - query.parameters.get(0);
		double dy = centerParameters.get(1) - query.parameters.get(1);
		return Math.sqrt(dx * dx + dy * dy) <= range;
	}

	@Override
	protected void constructSignature() {
		signature0 = new HashSet<>();
		signature = signature0;
		for (PatternInstance pi : elements) {
			ArrayList<Double> piParam = pi.parameters;
			ArrayList<Double> relParam = new ArrayList<>();
			for (int i = 0; i < piParam.size(); i++) {
				double param = piParam.get(i) - centerParameters.get(i);
				if (param == 0) // convert possible -0 to +0
					param = 0.0d;
				relParam.add(param);
			}
			signature0.add(new Pair<>(pi.type, relParam));
		}
	}

	@Override
	public void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI) {

		boolean hasA = elements.remove(pia);
		if (hasA) {
			ArrayList<Double> relParam = getRelParam(pia.parameters);
			signature0.remove(new Pair<>(pia.type, relParam));
		}

		boolean hasB = elements.remove(pib);
		if (hasB) {
			ArrayList<Double> relParam = getRelParam(pib.parameters);
			signature0.remove(new Pair<>(pib.type, relParam));
		}

		if (hasA || hasB) {
			elements.add(newPI);
			ArrayList<Double> relParam = getRelParam(newPI.parameters);
			signature0.add(new Pair<>(newPI.type, relParam));
		}
	}

	/**
	 * Convert the input parameters to a list of contextual parameters relative
	 * to the center parameters.
	 * 
	 * @param piParam
	 * @return
	 */
	protected ArrayList<Double> getRelParam(ArrayList<Double> piParam) {
		ArrayList<Double> relParam = new ArrayList<>();
		for (int i = 0; i < piParam.size(); i++) {
			double param = piParam.get(i) - centerParameters.get(i);
			if (param == 0) // convert possible -0 to +0
				param = 0.0d;
			relParam.add(param);
		}
		return relParam;
	}
}
