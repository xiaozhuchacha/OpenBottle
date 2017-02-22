package aog.learn.bc;

import aog.sample.PatternInstance;

/**
 * A context that is always empty.
 * 
 * @author Kewei Tu
 * 
 */
public class DefaultContext extends CompositionalContext {

	public DefaultContext() {
	}

	public DefaultContext(PatternInstance pi) {
		super(pi);
	}

	public DefaultContext(Context c) {
		if (c instanceof DefaultContext) {
			constructSignature();
		} else {
			System.err.println("[DefaultContext] type error.");
			System.exit(1);
		}
	}

	@Override
	protected boolean isContainValidForCompositePI() {
		return true;
	}

	@Override
	protected void constructSignature() {
		signature = Boolean.TRUE;
	}

	@Override
	public void updateWithBigramReduction(PatternInstance pia,
			PatternInstance pib, PatternInstance newPI) {
	}

	@Override
	protected boolean contains(PatternInstance query) {
		return false;
	}

	@Override
	protected Context getBigramContext2(PatternInstance pi1,
			CompositionalContext c1, PatternInstance pi2,
			CompositionalContext c2) {
		return this;
	}

}
