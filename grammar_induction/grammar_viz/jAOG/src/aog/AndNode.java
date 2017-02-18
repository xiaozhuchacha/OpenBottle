package aog;

import java.util.ArrayList;

import aog.relation.ICombiner;
import aog.relation.IRelation;

/**
 * An AND node.
 * <p>
 * By default, the parameters of an AND instance is the average of the
 * parameters of its two child instances.
 * 
 * @author Kewei Tu
 * 
 */
public class AndNode extends Pattern {

	public ArrayList<Pattern> children;

	// /**
	// * For each child, a relation is specified between this AND node and the
	// * child.
	// */
	// public ArrayList<IRelation> relations;

	/**
	 * The relation between this AND node and its children, represented by an
	 * ICombiner.
	 */
	public ICombiner pcRelation;

	/**
	 * The relations between children.
	 */
	public IRelation[][] ccRelations;

	public AndNode() {
		super();
		children = new ArrayList<>();
	}

	public AndNode(ArrayList<Pattern> children, IRelation[][] ccRelations,
			ICombiner pcRelation) {
		super();

		this.children = children;
		for (Pattern chld : children) {
			chld.parents.add(this);
		}

		this.ccRelations = ccRelations;
		this.pcRelation = pcRelation;
	}

	public void addChild(Pattern child) {
		children.add(child);
		child.parents.add(this);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("AND#" + id + "[");
		for (Pattern p : children) {
			if (p instanceof TerminalPattern)
				sb.append(p + ",");
			else
				sb.append("#" + p.id + ",");
		}
		sb.delete(sb.length() - 1, sb.length());
		sb.append("]");
		return sb.toString();
	}
}
