package aog.relation;

/**
 * A relation type is negatable if for any relation r1, there exists exactly one
 * relation r2, both belonging to the relation type, such that for any two
 * elements a, b, r1(a,b) <=> r2(b,a).
 * 
 * @author Kewei Tu
 * 
 */
public interface INegatableRelation extends IRelation {
	public INegatableRelation negate();
}
