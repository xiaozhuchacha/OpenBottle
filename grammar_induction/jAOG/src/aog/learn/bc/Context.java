package aog.learn.bc;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import aog.sample.PatternInstance;
import aog.sample.Sample;

/**
 * The context of a pattern instance or bigram. A pattern instance is in the
 * context iff. {@link #contains(PatternInstance)} returns true.
 * 
 * Context should be symmetrical, i.e., a is in the context of b iff. b is in
 * the context of a.
 * 
 * Subclasses must provide three constructors: one with no parameter for an
 * empty context, one with a PatternInstance object for a terminal context, and
 * one with a Context object for a signature-copy context. See the static
 * constructor helper methods of this class.
 * 
 * Note: For the use in aog.learn.bc.GrammarLearner, please extend
 * CompositionalContext instead of extending this class directly. There's some
 * issues in the current learning code s.t. non-compositional contexts would
 * lead to errors. One such issue: suppose A is in the context of B and vice
 * versa, then B is reduced to C; for non-compositional context it is possible
 * that C is not in the context of A, so later when A is reduced, the context of
 * B will not be updated to reflect the reduction (because currently we only
 * update the contexts of 1) instances in the context of the to-be-reduced
 * instances, and 2) the direct and indirect children of such instances).
 * 
 * @author Kewei Tu
 * 
 */
public abstract class Context {

	/**
	 * The set of pattern instances in this context
	 */
	protected Set<PatternInstance> elements;

	/**
	 * The signature of this context, which typically contains the types and
	 * relative parameters of the pattern instances in this context. Subclasses
	 * must specify its type.
	 */
	protected Object signature;

	/**
	 * The parameters of the center of this context.
	 */
	protected ArrayList<Double> centerParameters;

	protected Context() {
		elements = new HashSet<>();
	}

	/**
	 * Constructor based on a pattern instance
	 * 
	 * @param pi
	 */
	public Context(PatternInstance pi) {
		centerParameters = pi.parameters;
		elements = new HashSet<>();

		Sample s = pi.containingSample;
		for (PatternInstance pi2 : s.elements) {
			if (pi != pi2 && this.contains(pi2)) {
				elements.add(pi2);
			}
		}
		constructSignature();
	}

	/**
	 * Update the signature given elements and centerParameters.
	 */
	protected abstract void constructSignature();

	/**
	 * Test whether the query instance should be in this context
	 * 
	 * @param query
	 * @return
	 */
	protected abstract boolean contains(PatternInstance query);

	@Override
	public int hashCode() {
		return signature.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Context) {
			Context c2 = (Context) obj;
			return signature.equals(c2.signature);
		} else
			return false;
	}

	@Override
	public String toString() {
		return "[" + hashCode() + "]";
	}

	/**
	 * The range or size of the context. Different subclasses may have different
	 * interpretations of the range.
	 */
	public static double range = 0;

	public static String contextType;
	private static Constructor<?> emptyContextConstructor;
	private static Constructor<?> terminalContextConstructor;
	private static Constructor<?> signatureContextConstructor;

	/**
	 * @return a new empty context of type {@link #contextType}.
	 */
	protected static Context newEmptyContext() {
		try {
			if (emptyContextConstructor == null) {
				emptyContextConstructor = Class.forName(contextType)
						.getConstructor((Class<?>[]) null);
			}
			return (Context) emptyContextConstructor.newInstance();
		} catch (SecurityException | ReflectiveOperationException
				| IllegalArgumentException e) {
			e.printStackTrace();
			System.exit(1);
		}

		return null;
	}

	/**
	 * @param pi
	 * @return a new terminal context of pi of type {@link #contextType}.
	 */
	protected static Context newTerminalContext(PatternInstance pi) {
		try {
			if (terminalContextConstructor == null) {
				terminalContextConstructor = Class.forName(contextType)
						.getConstructor(
								new Class<?>[] { PatternInstance.class });
			}
			return (Context) terminalContextConstructor.newInstance(pi);
		} catch (SecurityException | ReflectiveOperationException
				| IllegalArgumentException e) {
			e.printStackTrace();
			System.exit(1);
		}

		return null;
	}

	/**
	 * @param signature
	 * @return a copy context as the key for HashMap when the original context
	 *         may be modified later. Only the signature of the context is
	 *         copied.
	 */
	protected static Context newProxyContextWithSignatureOf(Context c) {
		try {
			if (signatureContextConstructor == null) {
				signatureContextConstructor = Class.forName(contextType)
						.getConstructor(new Class<?>[] { Context.class });
			}
			return (Context) signatureContextConstructor.newInstance(c);
		} catch (SecurityException | ReflectiveOperationException
				| IllegalArgumentException e) {
			e.printStackTrace();
			System.exit(1);
		}

		return null;
	}
}
