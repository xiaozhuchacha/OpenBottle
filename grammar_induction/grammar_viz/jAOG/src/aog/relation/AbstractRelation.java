package aog.relation;

import aog.sample.PatternInstance;

/**
 * A relation between two patterns.
 * 
 * @author Kewei Tu
 * 
 */
public abstract class AbstractRelation implements IRelation {
	@Override
	public boolean hasRelation(PatternInstance a, PatternInstance b) {
		if (a.containingSample != b.containingSample)
			return false;
		else
			return hasRelation(a.type, a.parameters, b.type, b.parameters);
	}

	public IRelation clone() {
		try {
			return (IRelation) super.clone();
		} catch (CloneNotSupportedException e) {
			// this shouldn't happen
			e.printStackTrace();
			return null;
		}
	}

	@Override
	abstract public boolean equals(Object obj);

	@Override
	abstract public int hashCode();
}
