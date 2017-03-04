package aog;

import java.util.ArrayList;

public class OrNode extends Pattern {
	public ArrayList<Pattern> children;

	/**
	 * may not be normalized
	 */
	public ArrayList<Double> weights;

	public OrNode() {
		super();
		children = new ArrayList<Pattern>();
		weights = new ArrayList<Double>();
	}

	public OrNode(ArrayList<Pattern> children, ArrayList<Double> weights) {
		super();
		addChildren(children, weights);
	}

	public void addChildren(ArrayList<Pattern> children,
			ArrayList<Double> weights) {
		this.children = children;
		this.weights = weights;
		for (Pattern chld : children) {
			chld.parents.add(this);
		}
	}

	public void addChild(Pattern child, double weight) {
		children.add(child);
		weights.add(weight);
		child.parents.add(this);
	}

	// /**
	// * Decrease the weight of the "s" child by 1. If then its weight becomes
	// 0,
	// * delete the child.
	// *
	// * @param s
	// */
	// public void decreaseChildWeight(Symbol s) {
	// int ind = children.indexOf(s);
	// double i = weights.get(ind)[0];
	// if (i == 1) {
	// children.remove(ind);
	// weights.remove(ind);
	// s.parents.remove(this);
	// } else {
	// weights.get(ind)[0]--;
	// }
	// } //this method shall not appear here

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("OR#" + id + "[");
		for (Pattern p : children) {
			if (p instanceof TerminalPattern)
				sb.append(p + ",");
			else
				sb.append("#" + p.id + ",");
		}
		sb.delete(sb.length() - 1, sb.length());
		sb.append("][");
		for (double w : weights) {
			sb.append(w + ",");
		}
		sb.delete(sb.length() - 1, sb.length());
		sb.append("]");
		return sb.toString();
	}
}
