package aog.app.sequence;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import aog.Pattern;
import aog.learn.bc.CompositionalContext;
import aog.learn.bc.Context;
import aog.sample.PatternInstance;
import aog.util.Pair;

/**
 * Instances within a range from the position range of the center instance.
 * 
 * @author Kewei Tu
 * 
 */
public class SequenceContext extends CompositionalContext {

	public SequenceContext() {
		super();
	}

	public SequenceContext(PatternInstance pi) {
		super(pi);
		signature = new Pair<ArrayList<Pattern>, ArrayList<Pattern>>(leftTypes,
				rightTypes);
	}

	public SequenceContext(Context c) {
		if (c instanceof SequenceContext) {
			SequenceContext sc = (SequenceContext) c;
			leftTypes = new ArrayList<>(sc.leftTypes);
			rightTypes = new ArrayList<>(sc.rightTypes);
			signature = new Pair<ArrayList<Pattern>, ArrayList<Pattern>>(
					leftTypes, rightTypes);
		} else {
			System.err.println("[SequenceContext] type error.");
			System.exit(1);
		}
	}

	@Override
	protected boolean isContainValidForCompositePI() {
		return true;
	}

	@Override
	protected boolean contains(PatternInstance query) {
		if (centerParameters.isEmpty() || query.parameters.isEmpty()) {
			System.out
					.println("[Warning] SequenceContext: no parameter found in the pattern instances.");
			return false;
		}

		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		double ps = query.parameters.get(0);
		double pe = query.parameters.get(1);
		return (p0s - pe <= range && p0s - pe > 0)
				|| (ps - p0e <= range && ps - p0e > 0);
	}

	protected ArrayList<PatternInstance> leftElements;
	protected ArrayList<PatternInstance> rightElements;
	protected ArrayList<Pattern> leftTypes;
	protected ArrayList<Pattern> rightTypes;

	@Override
	protected void constructSignature() {
		ArrayList<PatternInstance> elementList = new ArrayList<>(elements);
		Collections.sort(elementList, new Comparator<PatternInstance>() {

			@Override
			public int compare(PatternInstance o1, PatternInstance o2) {
				int s1 = o1.parameters.get(0).intValue();
				int s2 = o2.parameters.get(0).intValue();
				return s1 - s2;
			}

		});

		int s0 = centerParameters.get(0).intValue();
		int firstRight = 0;
		while (firstRight < elementList.size()
				&& elementList.get(firstRight).parameters.get(0) < s0)
			firstRight++;

		leftElements = new ArrayList<>((int) range);
		rightElements = new ArrayList<>((int) range);
		leftTypes = new ArrayList<>((int) range);
		rightTypes = new ArrayList<>((int) range);
		signature = new Pair<ArrayList<Pattern>, ArrayList<Pattern>>(leftTypes,
				rightTypes);
		for (int i = 0; i < elementList.size(); i++) {
			PatternInstance pi = elementList.get(i);

			int relPos = i - firstRight;
			if (relPos >= 0) {
				rightElements.add(pi);
				rightTypes.add(pi.type);
			} else {
				leftElements.add(0, pi);
				leftTypes.add(0, pi.type);
			}
		}
	}

	@Override
	public void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI) {
		boolean hasA = elements.remove(pia);
		boolean hasB = elements.remove(pib);

		if (!hasA && !hasB)
			return;

		elements.add(newPI);

		PatternInstance pi = hasA ? pia : pib;
		if (leftElements.contains(pi)) {
			if (hasA && hasB) {
				int posA = leftElements.indexOf(pia);
				leftElements.set(posA, newPI);
				leftTypes.set(posA, newPI.type);

				int posB = leftElements.indexOf(pib);
				leftElements.remove(posB);
				leftTypes.remove(posB);
			} else { // hasA ^ hasB
				int pos = leftElements.size() - 1;
				assert leftElements.get(pos) == pi;
				leftElements.set(pos, newPI);
				leftTypes.set(pos, newPI.type);
			}

			// try adding new elements into the context
			int nToAdd = (int) (range - leftElements.size());
			PatternInstance edge = leftElements.get(leftElements.size() - 1);
			int pos = edge.parameters.get(0).intValue();
			for (int i = 0; i < nToAdd; i++) {
				boolean found = false;
				for (PatternInstance pi2 : edge.containingSample.elements) {
					int p = pi2.parameters.get(1).intValue();
					if (p + 1 == pos) {
						leftElements.add(pi2);
						elements.add(pi2);
						leftTypes.add(pi2.type);
						pos = pi2.parameters.get(0).intValue();
						found = true;
						break;
					}
				}
				if (!found)
					break;
			}
		} else {
			if (hasA && hasB) {
				int posA = rightElements.indexOf(pia);
				rightElements.set(posA, newPI);
				rightTypes.set(posA, newPI.type);

				int posB = rightElements.indexOf(pib);
				rightElements.remove(posB);
				rightTypes.remove(posB);
			} else { // hasA ^ hasB
				int pos = rightElements.size() - 1;
				assert rightElements.get(pos) == pi;
				rightElements.set(pos, newPI);
				rightTypes.set(pos, newPI.type);
			}

			// try adding new elements into the context
			int nToAdd = (int) (range - rightElements.size());
			PatternInstance edge = rightElements.get(rightElements.size() - 1);
			int pos = edge.parameters.get(1).intValue();
			for (int i = 0; i < nToAdd; i++) {
				boolean found = false;
				for (PatternInstance pi2 : edge.containingSample.elements) {
					int p = pi2.parameters.get(0).intValue();
					if (p - 1 == pos) {
						rightElements.add(pi2);
						elements.add(pi2);
						rightTypes.add(pi2.type);
						pos = pi2.parameters.get(1).intValue();
						found = true;
						break;
					}
				}
				if (!found)
					break;
			}
		}
	}
}
