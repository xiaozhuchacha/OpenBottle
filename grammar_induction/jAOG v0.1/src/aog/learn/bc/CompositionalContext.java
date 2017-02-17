package aog.learn.bc;

import java.util.ArrayList;
import java.util.HashSet;

import aog.relation.ICombiner;
import aog.sample.PatternInstance;

/**
 * A pattern instance A is in the compositional context of a composite pattern
 * instance B iff. A is in the context of at least one component of B.
 * <p>
 * Given a composite pattern instance, its context shall be generated by calling
 * {@link #getBigramContext(PatternInstance, Context, PatternInstance, Context)}
 * . Subclasses should define a boolean method "isContainValidForCompositePI" to
 * indicate whether the context returned by the constructor with a
 * PatternInstance argument is valid.
 * 
 * @author Kewei Tu
 * 
 */
public abstract class CompositionalContext extends Context {

	/**
	 * Whether it is valid to call the constructor with a PatternInstance
	 * argument (which in turn calls contains()) to generate the context of a
	 * composite pattern instance.
	 */
	protected boolean isContainValidForCompositePI() {
		return false;
	}

	/**
	 * May not be valid, depending on {@link #containsIsValidForCompositePI}
	 * 
	 * @param pi
	 */
	public CompositionalContext(PatternInstance pi) {
		super(pi);

		// must be a terminal pi if contain() is not valid for composite pi
		if (!isContainValidForCompositePI() && !pi.children.isEmpty()) {
			System.err
					.println("contains() is not valid for context construction of "
							+ "composite pattern instances here!");
			System.exit(1);
		}
	}

	public CompositionalContext() {
		super();
	}

	/**
	 * The default combiner used in getBigramContext()
	 */
	protected static ICombiner defaultCombiner;

	/**
	 * Return the context of the bigram of pi1 and pi2. The new context does not
	 * contain either pi1 or pi2. The new context has the same type as c1 and
	 * c2.
	 * 
	 * @param pi1
	 * @param c1
	 * @param pi2
	 * @param c2
	 * @return
	 */
	public static Context getBigramContext(PatternInstance pi1,
			CompositionalContext c1, PatternInstance pi2,
			CompositionalContext c2) {
		/*
		 * Since static methods cannot be overridden, we do the work in a
		 * non-static method that can be overridden by subclasses.
		 */
		return c1.getBigramContext2(pi1, c1, pi2, c2);
	}

	/**
	 * The actual method that does the work for
	 * {@link #getBigramContext(PatternInstance, Context, PatternInstance, Context)}
	 * .
	 * 
	 * @param pi1
	 * @param c1
	 * @param pi2
	 * @param c2
	 * @return
	 */
	protected Context getBigramContext2(PatternInstance pi1,
			CompositionalContext c1, PatternInstance pi2,
			CompositionalContext c2) {
		if (pi1.containingSample != pi2.containingSample)
			return null;

		if (c1 == null || c2 == null)
			return null;

		Context c3 = Context.newEmptyContext();
		ArrayList<ArrayList<Double>> paramOfChildren = new ArrayList<>();
		paramOfChildren.add(pi1.parameters);
		paramOfChildren.add(pi2.parameters);
		c3.centerParameters = defaultCombiner
				.getParentFromChildren(paramOfChildren);

		c3.elements = new HashSet<>(c1.elements);
		c3.elements.remove(pi2);
		c3.elements.addAll(c2.elements);
		c3.elements.remove(pi1);
		c3.constructSignature();

		return c3;
	}

	/**
	 * Update this context when pia and pib is merged into newPI. It shall be
	 * guaranteed that the instance to which this context belongs has no overlap
	 * with the input instances.
	 * 
	 * @param pia
	 * @param pib
	 * @param newPI
	 */
	public abstract void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI);
}
