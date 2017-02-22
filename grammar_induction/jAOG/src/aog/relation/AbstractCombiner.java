package aog.relation;

/**
 * The common superclass of all ICombiner classes.
 * 
 * @author Kewei Tu
 * 
 */
public abstract class AbstractCombiner implements ICombiner {

	@Override
	public ICombiner clone() {
		try {
			return (ICombiner) super.clone();
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
