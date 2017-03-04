package aog.app.vectorseq;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import aog.Pattern;
import aog.learn.bc.CompositionalContext;
import aog.learn.bc.Context;
import aog.sample.PatternInstance;

/**
 * The instances whose position range overlaps with the center range. The
 * signature of this context divides such instances into 4 categories: before,
 * after, contained, other (including partial overlapping and containing)
 * 
 * @author Kewei Tu
 * 
 */
public class VectorSeqContext2 extends CompositionalContext {

	protected Set<Pattern> leftTypes, rightTypes, containedTypes, otherTypes;
	protected Set<PatternInstance> leftElements, rightElements,
			containedElements, otherElements;

	public VectorSeqContext2() {
		super();
	}

	public VectorSeqContext2(PatternInstance pi) {
		super(pi);
		createSignatureFromFields();
	}

	public VectorSeqContext2(Context c) {
		if (c instanceof VectorSeqContext2) {
			VectorSeqContext2 vsc = (VectorSeqContext2) c;
			leftTypes = new HashSet<Pattern>(vsc.leftTypes);
			rightTypes = new HashSet<Pattern>(vsc.rightTypes);
			containedTypes = new HashSet<Pattern>(vsc.containedTypes);
			otherTypes = new HashSet<Pattern>(vsc.otherTypes);

			createSignatureFromFields();
		} else {
			System.err.println("[VectorSeqContext2] type error.");
			System.exit(1);
		}
	}

	@Override
	protected boolean isContainValidForCompositePI() {
		return true;
	}

	protected void createSignatureFromFields() {
		ArrayList<Set<Pattern>> signature0 = new ArrayList<Set<Pattern>>();
		signature0.add(leftTypes);
		signature0.add(rightTypes);
		signature0.add(containedTypes);
		signature0.add(otherTypes);
		signature = signature0;
	}

	@Override
	protected void constructSignature() {
		leftTypes = new HashSet<Pattern>();
		rightTypes = new HashSet<Pattern>();
		containedTypes = new HashSet<Pattern>();
		otherTypes = new HashSet<Pattern>();
		leftElements = new HashSet<PatternInstance>();
		rightElements = new HashSet<PatternInstance>();
		containedElements = new HashSet<PatternInstance>();
		otherElements = new HashSet<PatternInstance>();

		for (PatternInstance pi : elements) {
			addElementToFields(pi);
		}

		createSignatureFromFields();
	}

	@Override
	protected boolean contains(PatternInstance query) {
		if (centerParameters.isEmpty() || query.parameters.isEmpty()) {
			System.out
					.println("[Warning] VectorSeqContext2: no parameter found in the pattern instances.");
			return false;
		}

		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		double ps = query.parameters.get(0);
		double pe = query.parameters.get(1);

		// neighbor?
		if (ps <= p0e + range && pe >= p0s - range)
			return true;
		else
			return false;
	}

	@Override
	public void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI) {
		// remove
		boolean hasA = elements.remove(pia);
		if (hasA) {
			removeElementFromFields(pia);
		}

		boolean hasB = elements.remove(pib);
		if (hasB) {
			removeElementFromFields(pib);
		}

		// add
		if (hasA || hasB) {
			elements.add(newPI);
			addElementToFields(newPI);
		}
	}

	protected void addElementToFields(PatternInstance pi) {
		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		double ps = pi.parameters.get(0);
		double pe = pi.parameters.get(1);
		if (p0s - pe <= range && p0s > pe) {
			leftElements.add(pi);
			leftTypes.add(pi.type);
		} else if (ps - p0e <= range && ps > p0e) {
			rightElements.add(pi);
			rightTypes.add(pi.type);
		} else if (ps >= p0s && pe <= p0e) {
			containedElements.add(pi);
			containedTypes.add(pi.type);
		} else {
			otherElements.add(pi);
			otherTypes.add(pi.type);
		}
	}

	protected void removeElementFromFields(PatternInstance pi) {
		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		double ps = pi.parameters.get(0);
		double pe = pi.parameters.get(1);
		if (p0s - pe <= range && p0s > pe) {
			removeElement(pi, leftElements, leftTypes);
		} else if (ps - p0e <= range && ps > p0e) {
			removeElement(pi, rightElements, rightTypes);
		} else if (ps >= p0s && pe <= p0e) {
			removeElement(pi, containedElements, containedTypes);
		} else {
			removeElement(pi, otherElements, otherTypes);
		}
	}

	protected void removeElement(PatternInstance pi,
			Set<PatternInstance> elements, Set<Pattern> types) {
		if (elements.remove(pi)) {
			Pattern p = pi.type;
			boolean duplicateType = false;
			for (PatternInstance pi2 : elements) {
				if (pi2.type == p) {
					duplicateType = true;
					break;
				}
			}
			if (!duplicateType)
				types.remove(p);
		}
	}

}
