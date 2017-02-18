package aog.learn.bc;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import aog.Pattern;
import aog.relation.INegatableRelation;
import aog.relation.INoisyRelation;
import aog.relation.IRelation;

/**
 * A bi-cluster of a BigramMatrix. Notice that this class can represent a
 * partial bi-cluster, i.e., not all the appearances in the full bi-cluster
 * (defined by the row-header/column-header set) are included.
 * 
 * @author Kewei Tu
 */
public class BiCluster {

	// /**
	// * The underlying bigram matrix of this bicluster
	// */
	// protected Matrix<Pattern, Pattern, BigramInstanceSet> matrix;

	// /**
	// * The row/column set for the bi-cluster
	// */
	// protected HashSet<Symbol> rowHeaders, colHeaders;

	/**
	 * The row/column set for the bi-cluster mapped to the corresponding
	 * row/column sum.
	 */
	protected HashMap<Pattern, int[]> rowSums = new HashMap<Pattern, int[]>();
	protected HashMap<Pattern, int[]> colSums = new HashMap<Pattern, int[]>();

	/**
	 * The sum of all the bi-cluster elements
	 */
	protected int sum = 0;

	protected IRelation relation;

	protected HashMap<Context, int[]> contextVector = new HashMap<Context, int[]>();

	/**
	 * The log posterior gain by extracting this bi-cluster to an AndNode
	 */
	protected double posteriorGain = 0;

	public BiCluster() {
	}

	public BiCluster(BiCluster bc) {
		sum = bc.sum;
		relation = bc.relation;
		posteriorGain = bc.posteriorGain;

		for (Entry<Pattern, int[]> e : bc.rowSums.entrySet()) {
			rowSums.put(e.getKey(), new int[] { e.getValue()[0] });
		}
		for (Entry<Pattern, int[]> e : bc.colSums.entrySet()) {
			colSums.put(e.getKey(), new int[] { e.getValue()[0] });
		}

		for (Entry<Context, int[]> e : bc.contextVector.entrySet()) {
			contextVector.put(e.getKey(), new int[] { e.getValue()[0] });
		}
	}

	/**
	 * Get the row-headers of the bi-cluster, along with their appearance
	 * numbers in the bi-cluster.
	 * 
	 * @param rowHeaders
	 *            an empty list where the row-headers are to be stored
	 * @param appNums
	 *            an empty list where the row-header appearance numbers are to
	 *            be stored
	 */
	public void getRows(List<Pattern> rowHeaders, List<Double> appNums) {
		for (Entry<Pattern, int[]> e : rowSums.entrySet()) {
			rowHeaders.add(e.getKey());
			appNums.add((double) e.getValue()[0]);
		}
	}

	/**
	 * Get the col-headers of the bi-cluster, along with their appearance
	 * numbers in the bi-cluster.
	 * 
	 * @param colHeaders
	 *            an empty list where the col-headers are to be stored
	 * @param appNums
	 *            an empty list where the col-header appearance numbers are to
	 *            be stored
	 */
	public void getCols(List<Pattern> colHeaders, List<Double> appNums) {
		for (Entry<Pattern, int[]> e : colSums.entrySet()) {
			colHeaders.add(e.getKey());
			appNums.add((double) e.getValue()[0]);
		}
	}

	@Override
	public String toString() {
		String re = "Rows:[";
		for (Pattern s : rowSums.keySet())
			re += s + ",";
		re += "] Cols:[";
		for (Pattern s : colSums.keySet())
			re += s + ",";
		re += "] Posterior Gain = " + posteriorGain;

		// if (ect != null) {
		// re += "\n\n";
		// final int sizelimit = 50;
		// ArrayList<Pattern> cs = new ArrayList<Pattern>(sizelimit);
		// int count = 0;
		// for (Pattern c : cols.keySet()) {
		// cs.add(c);
		// count++;
		// if (count == sizelimit)
		// break;
		// }
		// for (Object ch : cs)
		// re += "\t" + ch;
		// if (cols.size() > sizelimit)
		// re += "\t(" + (cols.size() - sizelimit) + " more cols omitted)";
		// count = 0;
		// for (Pattern r : rows.keySet()) {
		// re += "\n" + r;
		// for (Pattern ch : cs) {
		// re += "\t";
		// Map<Context, Collection<Element>> ectrow = ect.rows
		// .get(new Expression(r, ch));
		// if (ectrow != null) {
		// int size = 0;
		// for (Collection<Element> cell : ectrow.values())
		// size += cell.size();
		// re += size;
		// }
		// }
		// count++;
		// if (count == sizelimit)
		// break;
		// }
		// if (rows.size() > sizelimit)
		// re += "\n(" + (rows.size() - sizelimit) + " more rows omitted)";
		// }

		return re;
	}

	/**
	 * Indicates whether the two biclusters are equivalent. Equivalence means
	 * either (a) the sets of rows and columns are the same respectively, and
	 * the relations are the same (in the case of noisy relations, approximately
	 * same); or (b) after one of the biclusters is transposed, they meet
	 * condition (a).
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof BiCluster) {
			BiCluster bc = (BiCluster) obj;
			if (rowSums.keySet().equals(bc.rowSums.keySet())
					&& colSums.keySet().equals(bc.colSums.keySet()))
				if (relation instanceof INoisyRelation) {
					INoisyRelation r1 = (INoisyRelation) relation;
					INoisyRelation r2 = (INoisyRelation) bc.relation;
					if (r1.approxEquals(r2))
						return true;
				} else {
					if (relation.equals(bc.relation))
						return true;
				}
			else if (rowSums.keySet().equals(bc.colSums.keySet())
					&& colSums.keySet().equals(bc.rowSums.keySet()))
				if (relation instanceof INoisyRelation) {
					INoisyRelation r1 = (INoisyRelation) relation;
					INoisyRelation r2 = (INoisyRelation) bc.relation;
					return r1.approxEquals((INoisyRelation) r2.negate());
				} else if (relation instanceof INegatableRelation) {
					INegatableRelation r2 = (INegatableRelation) bc.relation;
					return relation.equals(r2.negate());
				} else
					return relation.equals(bc.relation);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return rowSums.keySet().hashCode() + colSums.keySet().hashCode();
		/*
		 * relation is not used in hashing because they are transposition
		 * sensitive.
		 */
	}

	/**
	 * @param bc
	 * @return If the two biclusters are overlapping (with or without
	 *         transposition), i.e., they have shared rows and columns and have
	 *         the same relation.
	 */
	public boolean conflicts(BiCluster bc) {
		if (!Collections.disjoint(rowSums.keySet(), bc.rowSums.keySet())
				&& !Collections.disjoint(colSums.keySet(), bc.colSums.keySet())) {
			if (relation instanceof INoisyRelation) {
				INoisyRelation r1 = (INoisyRelation) relation;
				INoisyRelation r2 = (INoisyRelation) bc.relation;
				if (r1.approxEquals(r2))
					return true;
			} else {
				if (relation.equals(bc.relation))
					return true;
			}
		} else if (!Collections.disjoint(rowSums.keySet(), bc.colSums.keySet())
				&& !Collections.disjoint(colSums.keySet(), bc.rowSums.keySet())) {
			if (relation instanceof INoisyRelation) {
				INoisyRelation r1 = (INoisyRelation) relation;
				INoisyRelation r2 = (INoisyRelation) bc.relation;
				return r1.approxEquals((INoisyRelation) r2.negate());
			} else if (relation instanceof INegatableRelation) {
				INegatableRelation r2 = (INegatableRelation) bc.relation;
				return relation.equals(r2.negate());
			} else
				return relation.equals(bc.relation);
		}
		return false;
	}

	public static Class<?> relationClass;
}
