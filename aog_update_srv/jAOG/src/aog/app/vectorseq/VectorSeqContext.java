package aog.app.vectorseq;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import aog.Pattern;
import aog.learn.bc.Context;
import aog.sample.PatternInstance;

/**
 * Instances near the center instance. See {@link #mode}. XXX: not tested, use
 * with caution
 * 
 * @author Kewei Tu
 * 
 */
public class VectorSeqContext extends Context {

	/**
	 * 0: only left/right immediate neighbors (within the range).<br>
	 * 1: #0 + the units with the same start/end positions as the center
	 */
	protected static int mode = 1;

	protected Set<Pattern> leftTypes, rightTypes, cooccurTypes;
	protected Set<PatternInstance> leftElements, rightElements,
			cooccurElements;

	public VectorSeqContext() {
		super();
	}

	public VectorSeqContext(PatternInstance pi) {
		super(pi);
		ArrayList<Set<Pattern>> signature0 = new ArrayList<Set<Pattern>>();
		signature0.add(leftTypes);
		signature0.add(rightTypes);
		signature0.add(cooccurTypes);
		signature = signature0;
	}

	public VectorSeqContext(Context c) {
		if (c instanceof VectorSeqContext) {
			VectorSeqContext vsc = (VectorSeqContext) c;

			leftTypes = new HashSet<Pattern>(vsc.leftTypes);
			rightTypes = new HashSet<Pattern>(vsc.rightTypes);
			cooccurTypes = new HashSet<Pattern>(vsc.cooccurTypes);

			ArrayList<Set<Pattern>> signature0 = new ArrayList<Set<Pattern>>();
			signature0.add(leftTypes);
			signature0.add(rightTypes);
			signature0.add(cooccurTypes);
			signature = signature0;
		} else {
			System.err.println("[VectorSeqContext] type error.");
			System.exit(1);
		}
	}

	@Override
	protected void constructSignature() {
		leftTypes = new HashSet<Pattern>();
		rightTypes = new HashSet<Pattern>();
		cooccurTypes = new HashSet<Pattern>();
		leftElements = new HashSet<PatternInstance>();
		rightElements = new HashSet<PatternInstance>();
		cooccurElements = new HashSet<PatternInstance>();

		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		for (PatternInstance pi : elements) {
			double ps = pi.parameters.get(0);
			double pe = pi.parameters.get(1);
			if (p0s - pe <= range && p0s > pe) {
				leftElements.add(pi);
				leftTypes.add(pi.type);
			} else if (ps - p0e <= range && ps > p0e) {
				rightElements.add(pi);
				rightTypes.add(pi.type);
			} else if (mode == 1 && ps == p0s && pe == p0e) {
				cooccurElements.add(pi);
				cooccurTypes.add(pi.type);
			} else
				assert false;
		}

		ArrayList<Set<Pattern>> signature0 = new ArrayList<Set<Pattern>>();
		signature0.add(leftTypes);
		signature0.add(rightTypes);
		signature0.add(cooccurTypes);
		signature = signature0;
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

	@Override
	protected boolean contains(PatternInstance query) {
		if (centerParameters.isEmpty() || query.parameters.isEmpty()) {
			System.out
					.println("[Warning] VectorSeqContext: no parameter found in the pattern instances.");
			return false;
		}

		double p0s = centerParameters.get(0);
		double p0e = centerParameters.get(1);
		double ps = query.parameters.get(0);
		double pe = query.parameters.get(1);

		// neighbor?
		if ((p0s - pe <= range && p0s > pe) || (ps - p0e <= range && ps > p0e))
			return true;

		// cooccur?
		if (mode == 1)
			if (ps == p0s && pe == p0e)
				return true;

		return false;
	}

}
