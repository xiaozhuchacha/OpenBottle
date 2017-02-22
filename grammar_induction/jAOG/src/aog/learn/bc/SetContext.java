package aog.learn.bc;

import java.util.HashSet;
import java.util.Set;

import aog.Pattern;
import aog.sample.PatternInstance;

/**
 * The set-context contains all the instances in the sample other than the
 * center instance. The signature is simply the set of types of the instances
 * and their parameters are ignored.
 * 
 * @author Kewei Tu
 * 
 */
public class SetContext extends CompositionalContext {

	protected Set<Pattern> signature0;

	public SetContext() {
	}

	public SetContext(PatternInstance pi) {
		super(pi);
		signature = signature0;
	}

	public SetContext(Context c) {
		if (c instanceof SetContext) {
			SetContext sc = (SetContext) c;
			signature0 = new HashSet<>(sc.signature0);
			signature = signature0;
		} else {
			System.err.println("[SetContext] type error.");
			System.exit(1);
		}
	}

	@Override
	protected boolean isContainValidForCompositePI() {
		return true;
	}

	@Override
	protected boolean contains(PatternInstance query) {
		return true;
	}

	@Override
	protected void constructSignature() {
		signature0 = new HashSet<>();
		signature = signature0;
		for (PatternInstance pi : elements) {
			signature0.add(pi.type);
		}
	}

	@Override
	public void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI) {
		elements.remove(pia);
		elements.remove(pib);
		elements.add(newPI);
		constructSignature();
	}

}
