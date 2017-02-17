package aog.learn.bc;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import aog.Pattern;
import aog.learn.bc.BigramMatrix.BigramCell;
import aog.learn.bc.BigramMatrix.BigramInstanceSet;
import aog.relation.IRelation;
import aog.sample.PatternInstance;
import aog.util.Pair;
import aog.util.Triple;
import aog.util.Util;

public class BiClusterFinder {

	protected static BigramMatrix bgm;

	protected static Pattern[] rowHeaders;
	protected static Pattern[] colHeaders;

	public static void setBigramMatrix(BigramMatrix bgm) {
		BiClusterFinder.bgm = bgm;
		rowHeaders = bgm.getRowHeaders().toArray(new Pattern[0]);
		colHeaders = bgm.getColHeaders().toArray(new Pattern[0]);
	}

	/**
	 * Try to find a set of biclusters from the bigram matrix, using the input
	 * as a seed. The biclustering algorithm is beam search with hill-climbing
	 * (via adding/removing row/col) that optimizes the posterior gain.
	 * <p>
	 * We guarantee that the resulting bicluster does not contain two
	 * conflicting bigrams. A conflict between two bigrams means the two sets of
	 * bigram appearances contain at least two appearances, one from each set,
	 * such that they involve the same pattern instance (i.e., either
	 * <a,b>-<a,c> or <b,a>-<c,a>).
	 * 
	 * @param rows0
	 * @param cols0
	 * @param rel0
	 *            The seed bicluster. It must be valid (only containing
	 *            rows/cols presented in {@link #bgm}), and must not contain
	 *            duplicate rows/cols.
	 * @return a set of bi-clusters that have the highest posterior gain. The
	 *         number of biclusters is determined by {@link #beamSize} and
	 *         {@link #beamRelativeThreshold}.
	 */
	public static Collection<BiCluster> biClustering(Collection<Pattern> rows0,
			Collection<Pattern> cols0, IRelation rel0) {

		if (bgm.isEmpty())
			return new ArrayList<BiCluster>();

		/* Preprocessing seed */
		// remove conflicts
		for (Iterator<Pattern> iter = rows0.iterator(); iter.hasNext();) {
			Pattern r = iter.next();
			boolean delRow = false;
			ArrayList<Pattern> colCache = new ArrayList<>();
			for (Pattern c : cols0) {
				BigramInstanceSet appswr = bgm.getCellWithRelation(r, c, rel0);
				if (appswr != null) {
					// check internal conflicts
					for (Pattern c2 : colCache) {
						Set<PatternInstance> s1 = appswr.ugInstSets.a;
						Set<PatternInstance> s2 = bgm.getCellWithRelation(r,
								c2, rel0).ugInstSets.a;
						if (!Collections.disjoint(s1, s2)) {
							delRow = true;
							break;
						}
					}
					if (delRow)
						break;

					colCache.add(c);
				}
			}
			if (delRow)
				iter.remove();
		}
		if (rows0.isEmpty())
			return new ArrayList<BiCluster>();
		for (Iterator<Pattern> iter = cols0.iterator(); iter.hasNext();) {
			Pattern c = iter.next();
			boolean delCol = false;
			ArrayList<Pattern> rowCache = new ArrayList<>();
			for (Pattern r : cols0) {
				BigramInstanceSet appswr = bgm.getCellWithRelation(r, c, rel0);
				if (appswr != null) {
					// check internal conflicts
					for (Pattern r2 : rowCache) {
						Set<PatternInstance> s1 = appswr.ugInstSets.b;
						Set<PatternInstance> s2 = bgm.getCellWithRelation(r2,
								c, rel0).ugInstSets.b;
						if (!Collections.disjoint(s1, s2)) {
							delCol = true;
							break;
						}
					}
					if (delCol)
						break;

					rowCache.add(r);
				}
			}
			if (delCol)
				iter.remove();
		}
		if (cols0.isEmpty())
			return new ArrayList<BiCluster>();

		// create seed bc
		BiCluster bc = new BiCluster();
		bc.relation = rel0;
		bc.sum = 0;
		for (Pattern r : rows0) {
			int rowsum = 0;
			for (Pattern c : cols0) {
				BigramInstanceSet appswr = bgm.getCellWithRelation(r, c, rel0);
				if (appswr != null) {
					rowsum += appswr.size();

					// construct ectColSums by the way
					for (Map.Entry<Context, int[]> e : appswr.contextVector
							.entrySet()) {
						int[] colsum = bc.contextVector.get(e.getKey());
						if (colsum == null) {
							colsum = new int[] { 0 };
							Context key = Context
									.newProxyContextWithSignatureOf(e.getKey());
							bc.contextVector.put(key, colsum);
						}
						colsum[0] += e.getValue()[0];
					}
				}
			}
			int[] tmp = { rowsum };
			bc.rowSums.put(r, tmp);
			bc.sum += rowsum;
		}
		for (Pattern c : cols0) {
			int colsum = 0;
			for (Pattern r : rows0) {
				BigramCell apps = bgm.getCell(r, c);
				if (apps != null)
					colsum += apps.sizeWithRelation(rel0);
			}
			int[] tmp = { colsum };
			bc.colSums.put(c, tmp);
		}

		// TODO if bc.relation is noisy, adjust it to better fit the current bc?

		// compute posterior gain
		double likelihoodGain = 0;
		likelihoodGain -= Util.xlogx(bc.sum) * 2;
		for (int[] i : bc.colSums.values())
			likelihoodGain += Util.xlogx(i[0]);
		for (int[] colsum : bc.contextVector.values()) {
			likelihoodGain += Util.xlogx(colsum[0]);
		}
		for (Map.Entry<Pattern, int[]> e : bc.rowSums.entrySet()) {
			likelihoodGain += Util.xlogx(e.getValue()[0]);

			for (Pattern c : bc.colSums.keySet()) {
				BigramInstanceSet cwr = bgm.getCellWithRelation(e.getKey(), c,
						bc.relation);
				if (cwr != null) {
					for (int[] i : cwr.contextVector.values()) {
						likelihoodGain -= Util.xlogx(i[0]);
					}
				}
			}
		}
		double priorGain = alpha
				* (dl2 * bc.sum + dl3 * bc.rowSums.size() * bc.colSums.size() - dl1
						* (bc.rowSums.size() + bc.colSums.size()));
		bc.posteriorGain = likelihoodGain + priorGain;

		// The main loop
		BiClusterFinder bcf = new BiClusterFinder(bc);
		ArrayList<BiClusterFinder> beam = new ArrayList<>();
		ArrayList<BiClusterFinder> newBeam = new ArrayList<>();
		ArrayList<BiClusterFinder> oldBeam = new ArrayList<>();
		newBeam.add(bcf);
		while (!beam.equals(newBeam)) {
			ArrayList<BiClusterFinder> tmp = oldBeam;
			oldBeam = beam;
			beam = newBeam;
			tmp.clear();
			newBeam = tmp;

			for (BiClusterFinder f : beam) {
				if (!oldBeam.contains(f)) {
					f.updateContribution();
					ArrayList<BiClusterFinder> s = f.oneStepHillClimbing();
					addIntoBeam(newBeam, s);
				}
			}
			addIntoBeam(newBeam, beam);
		}

		// post-processing
		ArrayList<BiCluster> ret = new ArrayList<>();
		for (BiClusterFinder f : beam) {
			bc = f.bc;
			ret.add(bc);

			/* Abnormality checking */
			ArrayList<Pair<Pattern, Pattern>> abnormals = new ArrayList<>();
			double mult = mcbcMaxRatioToPrediction / bc.sum;
			for (Map.Entry<Pattern, int[]> r : bc.rowSums.entrySet()) {
				for (Map.Entry<Pattern, int[]> c : bc.colSums.entrySet()) {
					int a = bgm.getCellSizeWithRelation(r.getKey(), c.getKey(),
							bc.relation);
					if (a > mult * r.getValue()[0] * c.getValue()[0])
						abnormals.add(new Pair<Pattern, Pattern>(r.getKey(), c
								.getKey()));
				}
			}
			if (!abnormals.isEmpty())
				System.err.println("Find " + abnormals.size()
						+ " abnormally high values in the bicluster.");

			// // bicluster coherence checking
			// double coherence1 = 0, coherence2 = 0;
			// coherence1 -= Util.xlogx(bc.sum);
			// coherence2 -= Util.xlogx(bc.sum);
			// for (int[] i : bc.colSums.values())
			// coherence1 += Util.xlogx(i[0]);
			// for (int[] colsum : bc.contextVector.values()) {
			// coherence2 += Util.xlogx(colsum[0]);
			// }
			// for (Map.Entry<Pattern, int[]> e : bc.rowSums.entrySet()) {
			// coherence1 += Util.xlogx(e.getValue()[0]);
			//
			// for (Pattern c : bc.colSums.keySet()) {
			// BigramInstanceSet cwr = bgm.getCellWithRelation(e.getKey(),
			// c, bc.relation);
			// if (cwr != null) {
			// for (int[] i : cwr.contextVector.values()) {
			// coherence2 -= Util.xlogx(i[0]);
			// }
			//
			// coherence1 -= Util.xlogx(cwr.size());
			// coherence2 += Util.xlogx(cwr.size());
			// }
			// }
			// }
			//
			// /* Compute posterior */
			// double likelihoodGain = coherence1 + coherence2;
			// double priorGain = alpha
			// * (dl2 * bc.sum + dl3 * bc.rowSums.size()
			// * bc.colSums.size() - dl1
			// * (bc.rowSums.size() + bc.colSums.size()));
			// bc.posteriorGain = likelihoodGain + priorGain;
		}

		return ret;
	}

	/**
	 * @param beam
	 *            The current beam in which items are sorted in descending order
	 * @param newItems
	 */
	protected static void addIntoBeam(ArrayList<BiClusterFinder> beam,
			ArrayList<BiClusterFinder> newItems) {
		if (newItems.isEmpty())
			return;

		for (BiClusterFinder f : newItems) {
			if (!beam.contains(f))
				beam.add(f);
		}

		Collections.sort(beam, new Comparator<BiClusterFinder>() {
			@Override
			public int compare(BiClusterFinder o1, BiClusterFinder o2) {
				double diff = o2.bc.posteriorGain - o1.bc.posteriorGain;
				if (diff != 0)
					return (int) Math.signum(diff);
				else {
					// break the equivalence to avoid potential infinite loop
					BiCluster bc1 = o1.bc;
					BiCluster bc2 = o2.bc;
					if (bc1.colSums.size() != bc2.colSums.size()) {
						return bc1.colSums.size() - bc2.colSums.size();
					} else if (!bc1.colSums.keySet().equals(
							bc2.colSums.keySet())) {
						ArrayList<Pattern> cols1 = new ArrayList<>(bc1.colSums
								.keySet());
						ArrayList<Pattern> cols2 = new ArrayList<>(bc2.colSums
								.keySet());
						for (int i = 0; i < cols1.size(); i++) {
							Pattern col1 = cols1.get(i);
							Pattern col2 = cols2.get(i);
							if (col1.id == col2.id)
								continue;
							else
								return col1.id - col2.id;
						}
					} else if (bc1.rowSums.size() != bc2.rowSums.size()) {
						return bc1.rowSums.size() - bc2.rowSums.size();
					} else if (!bc1.rowSums.keySet().equals(
							bc2.rowSums.keySet())) {
						ArrayList<Pattern> rows1 = new ArrayList<>(bc1.rowSums
								.keySet());
						ArrayList<Pattern> rows2 = new ArrayList<>(bc2.rowSums
								.keySet());
						for (int i = 0; i < rows1.size(); i++) {
							Pattern row1 = rows1.get(i);
							Pattern row2 = rows2.get(i);
							if (row1.id == row2.id)
								continue;
							else
								return row1.id - row2.id;
						}
					}
					assert false;
					return 0;
				}
			}
		});

		double threshold = beam.get(0).bc.posteriorGain - beamRelativeThreshold;
		for (int i = beam.size() - 1; i >= 1; i--) {
			if (i >= beamSize || beam.get(i).bc.posteriorGain < threshold)
				beam.remove(i);
		}
	}

	protected static void addIntoBeam(
			ArrayList<Triple<Boolean, Integer, Double>> bestSteps,
			boolean isRow, int index, double contribution) {
		int i = Collections.binarySearch(bestSteps, contribution,
				new Comparator<Object>() {
					@SuppressWarnings("unchecked")
					@Override
					public int compare(Object o1, Object o2) {
						double c1 = o1 instanceof Double ? (Double) o1
								: ((Triple<Boolean, Integer, Double>) o1).c;
						double c2 = o2 instanceof Double ? (Double) o2
								: ((Triple<Boolean, Integer, Double>) o2).c;
						return (int) Math.signum(c2 - c1);
					}
				});
		i = i < 0 ? -i - 1 : i;
		if (i < beamSize) {
			bestSteps.add(i, new Triple<>(isRow, index, contribution));
			double threshold = bestSteps.get(0).c - beamRelativeThreshold;
			for (int j = bestSteps.size() - 1; j >= 1; j--) {
				if (j >= beamSize || bestSteps.get(j).c < threshold)
					bestSteps.remove(j);
			}
		}
	}

	public static void setParameters(double alpha, double dl1, double dl2,
			double dl3, double mcbcMaxRatioToPrediction, int beamSize,
			double beamRelativeThreshold, boolean allowConflict) {
		BiClusterFinder.alpha = alpha;
		BiClusterFinder.dl1 = dl1;
		BiClusterFinder.dl2 = dl2;
		BiClusterFinder.dl3 = dl3;
		BiClusterFinder.mcbcMaxRatioToPrediction = mcbcMaxRatioToPrediction;

		BiClusterFinder.beamSize = beamSize;
		BiClusterFinder.beamRelativeThreshold = beamRelativeThreshold;

		BiClusterFinder.allowConflict = allowConflict;
	}

	public static double mcbcMaxRatioToPrediction;
	public static double alpha;
	public static double dl1;
	public static double dl2;
	public static double dl3;

	public static boolean allowConflict;

	public static int beamSize;
	public static double beamRelativeThreshold;

	/*
	 * non-static members
	 */
	protected BiCluster bc;

	protected double[] r_contributions = new double[rowHeaders.length];
	protected double[] c_contributions = new double[colHeaders.length];
	protected int[] r_sums = new int[rowHeaders.length];
	protected int[] c_sums = new int[colHeaders.length];
	protected ArrayList<Map<Context, int[]>> r_EctColSums = new ArrayList<Map<Context, int[]>>(
			rowHeaders.length);
	protected ArrayList<Map<Context, int[]>> c_EctColSums = new ArrayList<Map<Context, int[]>>(
			colHeaders.length);

	/*
	 * used to identify conflicts 1) within a row/col to be added into the
	 * bicluster (internal); 2) between a row/col to be added and the bicluster
	 * (external).
	 */
	protected ArrayList<Collection<Pattern[]>> r_int_conflicts = new ArrayList<>(
			rowHeaders.length); // 2 cols in each conflict
	protected ArrayList<Collection<Pair<Pattern, Pattern>>> r_ext_conflicts = new ArrayList<>(
			rowHeaders.length); // <r,c> in bc for each conflict
	protected ArrayList<Collection<Pattern[]>> c_int_conflicts = new ArrayList<>(
			colHeaders.length); // 2 rows in each conflict
	protected ArrayList<Collection<Pair<Pattern, Pattern>>> c_ext_conflicts = new ArrayList<>(
			colHeaders.length); // <r,c> in bc for each conflict

	protected BiClusterFinder parentFinder;
	protected boolean validFields;

	protected BiClusterFinder(BiCluster seed) {
		bc = seed;

		// initialize the contrib/conflict of all possible steps
		for (int i = 0; i < rowHeaders.length; i++) {
			Pattern s = rowHeaders[i];
			boolean inBC = bc.rowSums.containsKey(s);

			// r_sums, conflicts
			ArrayList<Pattern[]> r_int_conflict = new ArrayList<>();
			r_int_conflicts.add(r_int_conflict);
			ArrayList<Pair<Pattern, Pattern>> r_ext_conflict = new ArrayList<>();
			r_ext_conflicts.add(r_ext_conflict);
			if (inBC) {
				r_sums[i] = bc.rowSums.get(s)[0];
			} else {
				r_sums[i] = 0;
				ArrayList<Pattern> nonEmptyCols = new ArrayList<>();
				for (Pattern sym : bc.colSums.keySet()) {
					BigramInstanceSet cwr = bgm.getCellWithRelation(s, sym,
							bc.relation);
					if (cwr != null) {
						r_sums[i] += cwr.size();
						nonEmptyCols.add(sym);
					}
				}

				// identify conflicts
				if (!allowConflict)
					for (int j = 0; j < nonEmptyCols.size(); j++) {
						Pattern c1 = nonEmptyCols.get(j);
						BigramInstanceSet bis = bgm.getCellWithRelation(s, c1,
								bc.relation);

						// internal
						for (int k = j + 1; k < nonEmptyCols.size(); k++) {
							Pattern c2 = nonEmptyCols.get(k);

							Set<PatternInstance> s1 = bis.ugInstSets.a;
							Set<PatternInstance> s2 = bgm.getCellWithRelation(
									s, c2, bc.relation).ugInstSets.a;
							if (!Collections.disjoint(s1, s2)) {
								r_int_conflict.add(new Pattern[] { c1, c2 });
							}
						}

						// external
						for (Pattern r2 : bc.rowSums.keySet()) {
							Set<PatternInstance> s1 = bis.ugInstSets.b;
							BigramInstanceSet appswr = bgm.getCellWithRelation(
									r2, c1, bc.relation);
							if (appswr != null) {
								Set<PatternInstance> s2 = appswr.ugInstSets.b;
								if (!Collections.disjoint(s1, s2)) {
									r_ext_conflict.add(new Pair<>(r2, c1));
								}
							}
						}
					}
			}

			// r_EctColSums
			Map<Context, int[]> newEctColSums = new HashMap<Context, int[]>();
			r_EctColSums.add(newEctColSums);

			// r_contributions: two coherence
			int sign = inBC ? -1 : 1;
			r_contributions[i] = 0;
			int rsum = r_sums[i];
			if (rsum != 0) {
				r_contributions[i] += sign * Util.xlogx(rsum);
				r_contributions[i] += 2 * Util.xlogx(bc.sum);
				r_contributions[i] -= 2 * Util.xlogx(sign * rsum + bc.sum);
				for (Map.Entry<Pattern, int[]> entry : bc.colSums.entrySet()) {
					BigramInstanceSet cwr = bgm.getCellWithRelation(s,
							entry.getKey(), bc.relation);
					if (cwr != null) {
						int a = cwr.size();
						int l = entry.getValue()[0];
						r_contributions[i] -= Util.xlogx(l);
						r_contributions[i] += Util.xlogx(sign * a + l);

						for (Map.Entry<Context, int[]> e : cwr.contextVector
								.entrySet()) {
							r_contributions[i] -= sign
									* Util.xlogx(e.getValue()[0]);

							int[] colsum = newEctColSums.get(e.getKey());
							if (colsum == null) {
								int[] tmp = { 0 };
								colsum = tmp;
								newEctColSums.put(e.getKey(), colsum);
							}
							colsum[0] += e.getValue()[0];
						}
					} // else it's "+=0"
				}
			} // else it's "+=0"

			for (Map.Entry<Context, int[]> e : newEctColSums.entrySet()) {
				int newc = e.getValue()[0];
				int[] c = bc.contextVector.get(e.getKey());
				if (c != null) {
					r_contributions[i] += Util.xlogx(c[0] + sign * newc);
					r_contributions[i] -= Util.xlogx(c[0]);
				} else
					r_contributions[i] += Util.xlogx(newc);
			}

			// r_contributions: prior change
			r_contributions[i] += alpha * sign
					* (-dl1 + dl2 * rsum + dl3 * bc.colSums.size());
		}
		for (int i = 0; i < colHeaders.length; i++) {
			Pattern s = colHeaders[i];
			boolean inBC = bc.colSums.containsKey(s);

			// c_sums, conflicts
			ArrayList<Pattern[]> c_int_conflict = new ArrayList<>();
			c_int_conflicts.add(c_int_conflict);
			ArrayList<Pair<Pattern, Pattern>> c_ext_conflict = new ArrayList<>();
			c_ext_conflicts.add(c_ext_conflict);
			if (inBC) {
				c_sums[i] = bc.colSums.get(s)[0];
			} else {
				c_sums[i] = 0;
				ArrayList<Pattern> nonEmptyRows = new ArrayList<>();
				for (Pattern sym : bc.rowSums.keySet()) {
					BigramInstanceSet cwr = bgm.getCellWithRelation(sym, s,
							bc.relation);
					if (cwr != null) {
						c_sums[i] += cwr.size();
						nonEmptyRows.add(sym);
					}
				}

				// identify conflicts
				if (!allowConflict)
					for (int j = 0; j < nonEmptyRows.size(); j++) {
						Pattern r1 = nonEmptyRows.get(j);
						BigramInstanceSet bis = bgm.getCellWithRelation(r1, s,
								bc.relation);

						// internal
						for (int k = j + 1; k < nonEmptyRows.size(); k++) {
							Pattern r2 = nonEmptyRows.get(k);

							Set<PatternInstance> s1 = bis.ugInstSets.b;
							Set<PatternInstance> s2 = bgm.getCellWithRelation(
									r2, s, bc.relation).ugInstSets.b;
							if (!Collections.disjoint(s1, s2)) {
								c_int_conflict.add(new Pattern[] { r1, r2 });
							}
						}

						// external
						for (Pattern c2 : bc.colSums.keySet()) {
							Set<PatternInstance> s1 = bis.ugInstSets.a;
							BigramInstanceSet appswr = bgm.getCellWithRelation(
									r1, c2, bc.relation);
							if (appswr != null) {
								Set<PatternInstance> s2 = appswr.ugInstSets.a;
								if (!Collections.disjoint(s1, s2)) {
									c_ext_conflict.add(new Pair<>(r1, c2));
								}
							}
						}
					}
			}

			// c_EctColSums
			Map<Context, int[]> newEctColSums = new HashMap<Context, int[]>();
			c_EctColSums.add(newEctColSums);

			// c_contributions: two coherence
			int sign = inBC ? -1 : 1;
			c_contributions[i] = 0;
			int csum = c_sums[i];
			if (csum != 0) {
				c_contributions[i] += sign * Util.xlogx(csum);
				c_contributions[i] += 2 * Util.xlogx(bc.sum);
				c_contributions[i] -= 2 * Util.xlogx(sign * csum + bc.sum);
				for (Map.Entry<Pattern, int[]> entry : bc.rowSums.entrySet()) {
					BigramInstanceSet cwr = bgm.getCellWithRelation(
							entry.getKey(), s, bc.relation);
					if (cwr != null) {
						int a = cwr.size();
						int l = entry.getValue()[0];
						c_contributions[i] -= Util.xlogx(l);
						c_contributions[i] += Util.xlogx(sign * a + l);

						for (Map.Entry<Context, int[]> e : cwr.contextVector
								.entrySet()) {
							c_contributions[i] -= sign
									* Util.xlogx(e.getValue()[0]);

							int[] colsum = newEctColSums.get(e.getKey());
							if (colsum == null) {
								int[] tmp = { 0 };
								colsum = tmp;
								newEctColSums.put(e.getKey(), colsum);
							}
							colsum[0] += e.getValue()[0];
						}
					} // else it's "+=0"
				}
			} // else it's "+=0"

			for (Map.Entry<Context, int[]> e : newEctColSums.entrySet()) {
				int newc = e.getValue()[0];
				int[] c = bc.contextVector.get(e.getKey());
				if (c != null) {
					c_contributions[i] += Util.xlogx(c[0] + sign * newc);
					c_contributions[i] -= Util.xlogx(c[0]);
				} else
					c_contributions[i] += Util.xlogx(newc);
			}

			// c_contributions: prior change
			c_contributions[i] += alpha * sign
					* (-dl1 + dl2 * csum + dl3 * bc.rowSums.size());
		}

		validFields = true;
	}

	protected BiClusterFinder(BiClusterFinder bcf, boolean nextStepIsRow,
			int nextStepIndex, double contribution) {

		parentFinder = bcf;
		validFields = false;

		// copy bc and next step
		bc = new BiCluster(bcf.bc);
		lastStepIsRow = nextStepIsRow;
		lastStepIndex = nextStepIndex;

		// update bc
		lastStepSymbol = lastStepIsRow ? rowHeaders[lastStepIndex]
				: colHeaders[lastStepIndex];
		lastStepIsAdding = lastStepIsRow ? !bc.rowSums
				.containsKey(lastStepSymbol) : !bc.colSums
				.containsKey(lastStepSymbol);
		int bSign = lastStepIsAdding ? 1 : -1;
		if (lastStepIsRow) {
			if (lastStepIsAdding) {
				int[] tmp = { bcf.r_sums[lastStepIndex] };
				bc.rowSums.put(lastStepSymbol, tmp);
			} else
				bc.rowSums.remove(lastStepSymbol);
			bc.sum += bSign * bcf.r_sums[lastStepIndex];
			for (Map.Entry<Pattern, int[]> entry : bc.colSums.entrySet()) {
				BigramCell c = bgm.getCell(lastStepSymbol, entry.getKey());
				int b = (c != null ? c.sizeWithRelation(bc.relation) : 0)
						* bSign;
				entry.getValue()[0] += b;
			}
			for (Map.Entry<Context, int[]> e : bcf.r_EctColSums.get(
					lastStepIndex).entrySet()) {
				int[] colsum = bc.contextVector.get(e.getKey());
				if (colsum == null) {
					colsum = new int[] { 0 };
					Context key = Context.newProxyContextWithSignatureOf(e
							.getKey());
					bc.contextVector.put(key, colsum);
				}
				colsum[0] += bSign * e.getValue()[0];
				if (colsum[0] == 0)
					bc.contextVector.remove(e.getKey());
			}
		} else {
			if (lastStepIsAdding) {
				int[] tmp = { bcf.c_sums[lastStepIndex] };
				bc.colSums.put(lastStepSymbol, tmp);
			} else
				bc.colSums.remove(lastStepSymbol);
			bc.sum += bSign * bcf.c_sums[lastStepIndex];
			for (Map.Entry<Pattern, int[]> entry : bc.rowSums.entrySet()) {
				BigramCell c = bgm.getCell(entry.getKey(), lastStepSymbol);
				int b = (c != null ? c.sizeWithRelation(bc.relation) : 0)
						* bSign;
				entry.getValue()[0] += b;
			}
			for (Map.Entry<Context, int[]> e : bcf.c_EctColSums.get(
					lastStepIndex).entrySet()) {
				int[] colsum = bc.contextVector.get(e.getKey());
				if (colsum == null) {
					colsum = new int[] { 0 };
					Context key = Context.newProxyContextWithSignatureOf(e
							.getKey());
					bc.contextVector.put(key, colsum);
				}
				colsum[0] += bSign * e.getValue()[0];
				if (colsum[0] == 0)
					bc.contextVector.remove(e.getKey());
			}
		}
		bc.posteriorGain += contribution;

		/*
		 * TODO update bc.relation if it's a noisy relation; then recompute all
		 * the contributions and conflicts
		 */
	}

	protected boolean lastStepIsRow;
	protected boolean lastStepIsAdding;
	protected Pattern lastStepSymbol;
	protected int lastStepIndex;

	/**
	 * @return a set of biclusters with the highest posterior gain after one
	 *         step of hill climbing from the current bicluster.
	 */
	protected ArrayList<BiClusterFinder> oneStepHillClimbing() {
		ArrayList<Triple<Boolean, Integer, Double>> bestSteps = new ArrayList<>();
		for (int i = 0; i < r_contributions.length; i++) {
			if (bc.rowSums.containsKey(rowHeaders[i]) && bc.rowSums.size() == 1)
				continue;
			if (r_sums[i] == 0 && !bc.rowSums.containsKey(rowHeaders[i]))
				continue;
			if (!allowConflict
					&& (!r_int_conflicts.get(i).isEmpty() || !r_ext_conflicts
							.get(i).isEmpty()))
				continue;
			addIntoBeam(bestSteps, true, i, r_contributions[i]);
		}
		for (int i = 0; i < c_contributions.length; i++) {
			if (bc.colSums.containsKey(colHeaders[i]) && bc.colSums.size() == 1)
				continue;
			if (c_sums[i] == 0 && !bc.colSums.containsKey(colHeaders[i]))
				continue;
			if (!allowConflict
					&& (!c_int_conflicts.get(i).isEmpty() || !c_ext_conflicts
							.get(i).isEmpty()))
				continue;
			addIntoBeam(bestSteps, false, i, c_contributions[i]);
		}

		ArrayList<BiClusterFinder> ret = new ArrayList<>();
		for (Triple<Boolean, Integer, Double> step : bestSteps) {
			BiClusterFinder f2 = new BiClusterFinder(this, step.a, step.b,
					step.c);
			ret.add(f2);
		}
		return ret;
	}

	protected void updateContribution() {
		if (validFields)
			return;

		copyFieldsFromParent();

		// update the contrib/conflict of all possible steps
		int bSign = lastStepIsAdding ? 1 : -1;
		// rows
		for (int i = 0; i < rowHeaders.length; i++) {
			// a for rowHeaders[i], b for nextStepSymbol
			int aSign = bc.rowSums.containsKey(rowHeaders[i]) ? -1 : 1;

			Collection<Pattern[]> r_int_conflict = r_int_conflicts.get(i);
			Collection<Pair<Pattern, Pattern>> r_ext_conflict = r_ext_conflicts
					.get(i);

			// update contrib/conflict
			if (lastStepIsRow) {
				if (rowHeaders[i] == lastStepSymbol) {
					r_contributions[i] = -r_contributions[i];
				} else {
					/*
					 * TODO Enhancement: If bgm is sparse, we can make this
					 * block more efficient. Currently, for each row we check
					 * each of the bi-cluster columns. Instead, we can go over
					 * each bi-cluster column and update the involved rows,
					 * which is faster because we don't have to go over all
					 * rows.
					 */

					// compute contribution change
					int aSum = r_sums[i];
					int bSum = r_sums[lastStepIndex] * bSign;
					if (aSum != 0 && bSum != 0) {
						r_contributions[i] += 2 * Util.xlogx(bc.sum);
						r_contributions[i] -= 2 * Util.xlogx(bc.sum + aSign
								* aSum);
						r_contributions[i] -= 2 * Util.xlogx(bc.sum - bSum);
						r_contributions[i] += 2 * Util.xlogx(bc.sum - bSum
								+ aSign * aSum);

						for (Map.Entry<Pattern, int[]> e : bc.colSums
								.entrySet()) {
							BigramCell c = bgm.getCell(rowHeaders[i],
									e.getKey());
							if (c != null) {
								int a = c.sizeWithRelation(bc.relation);
								c = bgm.getCell(lastStepSymbol, e.getKey());
								if (c != null) {
									int b = c.sizeWithRelation(bc.relation)
											* bSign;
									int l = e.getValue()[0];
									r_contributions[i] -= Util.xlogx(l);
									r_contributions[i] += Util.xlogx(aSign * a
											+ l);
									r_contributions[i] += Util.xlogx(l - b);
									r_contributions[i] -= Util.xlogx(aSign * a
											+ l - b);
								} // else it's +=0
							} // else it's +=0
						}

						Map<Context, int[]> nextStepEctColSums = r_EctColSums
								.get(lastStepIndex);
						for (Map.Entry<Context, int[]> e : r_EctColSums.get(i)
								.entrySet()) {
							int[] tmp = nextStepEctColSums.get(e.getKey());
							if (tmp != null) {
								int cb = tmp[0] * bSign;
								tmp = bc.contextVector.get(e.getKey());
								int c = tmp == null ? 0 : tmp[0];
								r_contributions[i] -= Util.xlogx(c);
								r_contributions[i] += Util.xlogx(aSign
										* e.getValue()[0] + c);
								r_contributions[i] += Util.xlogx(c - cb);
								r_contributions[i] -= Util.xlogx(aSign
										* e.getValue()[0] + c - cb);
							} // else it's +=0
						}
					} // else it's not changed

					// update conflict
					if (!allowConflict)
						// only need to check external conflict
						if (aSign == 1 && r_sums[i] != 0) {
							if (lastStepIsAdding) {
								for (Pattern c : bc.colSums.keySet()) {
									BigramInstanceSet bis = bgm
											.getCellWithRelation(rowHeaders[i],
													c, bc.relation);
									if (bis != null) {
										BigramInstanceSet bis2 = bgm
												.getCellWithRelation(
														lastStepSymbol, c,
														bc.relation);
										if (bis2 != null) {
											if (!Collections.disjoint(
													bis.ugInstSets.b,
													bis2.ugInstSets.b)) {
												r_ext_conflict.add(new Pair<>(
														lastStepSymbol, c));
											}
										}
									}
								}
							} else {
								for (Iterator<Pair<Pattern, Pattern>> iter = r_ext_conflict
										.iterator(); iter.hasNext();) {
									Pair<Pattern, Pattern> conf = iter.next();
									if (conf.a == lastStepSymbol)
										iter.remove();
								}
							}
						}
				}
			} else {
				// compute contribution change
				int aSum = r_sums[i];
				int bSum = c_sums[lastStepIndex];
				int x = 0;
				BigramInstanceSet cwr = bgm.getCellWithRelation(rowHeaders[i],
						lastStepSymbol, bc.relation);
				if (cwr != null)
					x = cwr.size();

				double dContrib = 0;

				// change 1
				/*
				 * the following if's are unnecessary but may accelerate the
				 * computation
				 */
				if (x != 0) {
					dContrib -= aSign * Util.xlogx(aSum);
					dContrib += aSign * Util.xlogx(aSum + bSign * x);
					dContrib -= bSign * Util.xlogx(bSum);
					dContrib += bSign * Util.xlogx(bSum + aSign * x);
				} // else it's "+=0"
				if (aSum != 0) {
					dContrib += 2 * Util.xlogx(bc.sum - bSign * bSum + aSign
							* aSum);
					dContrib -= 2 * Util.xlogx(bc.sum - bSign * bSum);
				} // else it's "+=0"
				if (aSum + bSign * x != 0) {
					dContrib += 2 * Util.xlogx(bc.sum);
					dContrib -= 2 * Util.xlogx(bc.sum + aSign * aSum + bSign
							* aSign * x);
				} // else it's "+=0"

				// update r_sum
				r_sums[i] += bSign * x;

				// change 2
				Map<Context, int[]> aEctColSums = r_EctColSums.get(i);
				Map<Context, int[]> bEctColSums = c_EctColSums
						.get(lastStepIndex);
				for (Map.Entry<Context, int[]> e : aEctColSums.entrySet()) {
					int[] tmp = bc.contextVector.get(e.getKey());
					int c = tmp == null ? 0 : tmp[0];
					tmp = bEctColSums.get(e.getKey());
					int cb = tmp == null ? 0 : tmp[0] * bSign;
					dContrib -= Util.xlogx(c - cb + aSign * e.getValue()[0]);
					dContrib += Util.xlogx(c - cb);
				}

				// update r_EctColSums
				if (cwr != null)
					for (Map.Entry<Context, int[]> e : cwr.contextVector
							.entrySet()) {
						int[] colsum = aEctColSums.get(e.getKey());
						if (colsum == null) {
							int[] tmp = { 0 };
							colsum = tmp;
							aEctColSums.put(e.getKey(), colsum);
						}
						colsum[0] += bSign * e.getValue()[0];
						if (colsum[0] == 0)
							aEctColSums.remove(e.getKey());

						// btw, change 2 cont'd
						dContrib -= aSign * bSign * Util.xlogx(e.getValue()[0]);
					}

				// change 2 cont'd
				for (Map.Entry<Context, int[]> e : aEctColSums.entrySet()) {
					int[] tmp = bc.contextVector.get(e.getKey());
					int c = tmp == null ? 0 : tmp[0];
					dContrib += Util.xlogx(c + aSign * e.getValue()[0]);
					dContrib -= Util.xlogx(c);
				}

				// change 3: prior change
				dContrib += alpha * bSign * (dl2 * aSign * x + dl3);

				// update contribution
				r_contributions[i] += dContrib;

				// update conflict
				if (!allowConflict)
					if (aSign == 1) { // not in bc
						if (lastStepIsAdding) {
							if (cwr != null) {
								// internal
								for (Pattern c : bc.colSums.keySet()) {
									if (c == lastStepSymbol)
										continue;
									BigramInstanceSet cwr2 = bgm
											.getCellWithRelation(rowHeaders[i],
													c, bc.relation);
									if (cwr2 != null) {
										if (!Collections.disjoint(
												cwr2.ugInstSets.a,
												cwr.ugInstSets.a)) {
											r_int_conflict.add(new Pattern[] {
													lastStepSymbol, c });
										}
									}
								}

								// external
								for (Pattern r : bc.rowSums.keySet()) {
									BigramInstanceSet cwr2 = bgm
											.getCellWithRelation(r,
													lastStepSymbol, bc.relation);
									if (cwr2 != null) {
										if (!Collections.disjoint(
												cwr2.ugInstSets.b,
												cwr.ugInstSets.b)) {
											r_ext_conflict.add(new Pair<>(r,
													lastStepSymbol));
										}
									}
								}
							}
						} else {
							// internal
							for (Iterator<Pattern[]> iter = r_int_conflict
									.iterator(); iter.hasNext();) {
								Pattern[] conf = iter.next();
								if (conf[0] == lastStepSymbol
										|| conf[1] == lastStepSymbol)
									iter.remove();
							}

							// external
							for (Iterator<Pair<Pattern, Pattern>> iter = r_ext_conflict
									.iterator(); iter.hasNext();) {
								Pair<Pattern, Pattern> conf = iter.next();
								if (conf.b == lastStepSymbol)
									iter.remove();
							}
						}
					}
			}
		}
		// cols
		for (int i = 0; i < colHeaders.length; i++) {
			// a for colHeaders[i], b for nextStepSymbol
			int aSign = bc.colSums.containsKey(colHeaders[i]) ? -1 : 1;

			Collection<Pattern[]> c_int_conflict = c_int_conflicts.get(i);
			Collection<Pair<Pattern, Pattern>> c_ext_conflict = c_ext_conflicts
					.get(i);

			// update contrib/conflict
			if (!lastStepIsRow) {
				if (colHeaders[i] == lastStepSymbol) {
					c_contributions[i] = -c_contributions[i];
				} else {
					/*
					 * TODO Enhancement: See the to-do enhancement above.
					 */

					// compute contribution change
					int aSum = c_sums[i];
					int bSum = c_sums[lastStepIndex] * bSign;
					if (aSum != 0 && bSum != 0) {
						c_contributions[i] += 2 * Util.xlogx(bc.sum);
						c_contributions[i] -= 2 * Util.xlogx(bc.sum + aSign
								* aSum);
						c_contributions[i] -= 2 * Util.xlogx(bc.sum - bSum);
						c_contributions[i] += 2 * Util.xlogx(bc.sum - bSum
								+ aSign * aSum);

						for (Map.Entry<Pattern, int[]> e : bc.rowSums
								.entrySet()) {
							BigramCell c = bgm.getCell(e.getKey(),
									colHeaders[i]);
							if (c != null) {
								int a = c.sizeWithRelation(bc.relation);
								c = bgm.getCell(e.getKey(), lastStepSymbol);
								if (c != null) {
									int b = c.sizeWithRelation(bc.relation)
											* bSign;
									int l = e.getValue()[0];
									c_contributions[i] -= Util.xlogx(l);
									c_contributions[i] += Util.xlogx(aSign * a
											+ l);
									c_contributions[i] += Util.xlogx(l - b);
									c_contributions[i] -= Util.xlogx(aSign * a
											+ l - b);
								} // else it's +=0
							} // else it's +=0
						}

						Map<Context, int[]> nextStepEctColSums = c_EctColSums
								.get(lastStepIndex);
						for (Map.Entry<Context, int[]> e : c_EctColSums.get(i)
								.entrySet()) {
							int[] tmp = nextStepEctColSums.get(e.getKey());
							if (tmp != null) {
								int cb = tmp[0] * bSign;
								tmp = bc.contextVector.get(e.getKey());
								int c = tmp == null ? 0 : tmp[0];
								c_contributions[i] -= Util.xlogx(c);
								c_contributions[i] += Util.xlogx(aSign
										* e.getValue()[0] + c);
								c_contributions[i] += Util.xlogx(c - cb);
								c_contributions[i] -= Util.xlogx(aSign
										* e.getValue()[0] + c - cb);
							} // else it's +=0
						}
					} // else it's not changed

					// update conflict
					if (!allowConflict)
						// only need to check external conflict
						if (aSign == 1 && c_sums[i] != 0) {
							if (lastStepIsAdding) {
								for (Pattern r : bc.rowSums.keySet()) {
									BigramInstanceSet bis = bgm
											.getCellWithRelation(r,
													colHeaders[i], bc.relation);
									if (bis != null) {
										BigramInstanceSet bis2 = bgm
												.getCellWithRelation(r,
														lastStepSymbol,
														bc.relation);
										if (bis2 != null) {
											if (!Collections.disjoint(
													bis.ugInstSets.a,
													bis2.ugInstSets.a)) {
												c_ext_conflict.add(new Pair<>(
														r, lastStepSymbol));
											}
										}
									}
								}
							} else {
								for (Iterator<Pair<Pattern, Pattern>> iter = c_ext_conflict
										.iterator(); iter.hasNext();) {
									Pair<Pattern, Pattern> conf = iter.next();
									if (conf.b == lastStepSymbol)
										iter.remove();
								}
							}
						}
				}
			} else {
				// compute contribution change
				int aSum = c_sums[i];
				int bSum = r_sums[lastStepIndex];
				int x = 0;
				BigramInstanceSet cwr = bgm.getCellWithRelation(lastStepSymbol,
						colHeaders[i], bc.relation);
				if (cwr != null)
					x = cwr.size();

				double dContrib = 0;

				// change 1
				/*
				 * the following if's are unnecessary but may accelerate the
				 * computation
				 */
				if (x != 0) {
					dContrib -= aSign * Util.xlogx(aSum);
					dContrib += aSign * Util.xlogx(aSum + bSign * x);
					dContrib -= bSign * Util.xlogx(bSum);
					dContrib += bSign * Util.xlogx(bSum + aSign * x);
				} // else it's "+=0"
				if (aSum != 0) {
					dContrib += 2 * Util.xlogx(bc.sum - bSign * bSum + aSign
							* aSum);
					dContrib -= 2 * Util.xlogx(bc.sum - bSign * bSum);
				} // else it's "+=0"
				if (aSum + bSign * x != 0) {
					dContrib += 2 * Util.xlogx(bc.sum);
					dContrib -= 2 * Util.xlogx(bc.sum + aSign * aSum + bSign
							* aSign * x);
				} // else it's "+=0"

				// update c_sum
				c_sums[i] += bSign * x;

				// change 2
				Map<Context, int[]> aEctColSums = c_EctColSums.get(i);
				Map<Context, int[]> bEctColSums = r_EctColSums
						.get(lastStepIndex);
				for (Map.Entry<Context, int[]> e : aEctColSums.entrySet()) {
					int[] tmp = bc.contextVector.get(e.getKey());
					int c = tmp == null ? 0 : tmp[0];
					tmp = bEctColSums.get(e.getKey());
					int cb = tmp == null ? 0 : tmp[0] * bSign;
					dContrib -= Util.xlogx(c - cb + aSign * e.getValue()[0]);
					dContrib += Util.xlogx(c - cb);
				}

				// update c_EctColSums
				if (cwr != null)
					for (Map.Entry<Context, int[]> e : cwr.contextVector
							.entrySet()) {
						int[] colsum = aEctColSums.get(e.getKey());
						if (colsum == null) {
							int[] tmp = { 0 };
							colsum = tmp;
							aEctColSums.put(e.getKey(), colsum);
						}
						colsum[0] += bSign * e.getValue()[0];
						if (colsum[0] == 0)
							aEctColSums.remove(e.getKey());

						// btw, change 2 cont'd
						dContrib -= aSign * bSign * Util.xlogx(e.getValue()[0]);
					}

				// change 2 cont'd
				for (Map.Entry<Context, int[]> e : aEctColSums.entrySet()) {
					int[] tmp = bc.contextVector.get(e.getKey());
					int c = tmp == null ? 0 : tmp[0];
					dContrib += Util.xlogx(c + aSign * e.getValue()[0]);
					dContrib -= Util.xlogx(c);
				}

				// change 3: prior change
				dContrib += alpha * bSign * (dl2 * aSign * x + dl3);

				// update contribution
				c_contributions[i] += dContrib;

				// update conflict
				if (!allowConflict)
					if (aSign == 1) { // not in bc
						if (lastStepIsAdding) {
							if (cwr != null) {
								// internal
								for (Pattern r : bc.rowSums.keySet()) {
									if (r == lastStepSymbol)
										continue;
									BigramInstanceSet cwr2 = bgm
											.getCellWithRelation(r,
													colHeaders[i], bc.relation);
									if (cwr2 != null) {
										if (!Collections.disjoint(
												cwr2.ugInstSets.b,
												cwr.ugInstSets.b)) {
											c_int_conflict.add(new Pattern[] {
													lastStepSymbol, r });
										}
									}
								}

								// external
								for (Pattern c : bc.colSums.keySet()) {
									BigramInstanceSet cwr2 = bgm
											.getCellWithRelation(
													lastStepSymbol, c,
													bc.relation);
									if (cwr2 != null) {
										if (!Collections.disjoint(
												cwr2.ugInstSets.a,
												cwr.ugInstSets.a)) {
											c_ext_conflict.add(new Pair<>(
													lastStepSymbol, c));
										}
									}
								}
							}
						} else {
							// internal
							for (Iterator<Pattern[]> iter = c_int_conflict
									.iterator(); iter.hasNext();) {
								Pattern[] conf = iter.next();
								if (conf[0] == lastStepSymbol
										|| conf[1] == lastStepSymbol)
									iter.remove();
							}

							// external
							for (Iterator<Pair<Pattern, Pattern>> iter = c_ext_conflict
									.iterator(); iter.hasNext();) {
								Pair<Pattern, Pattern> conf = iter.next();
								if (conf.a == lastStepSymbol)
									iter.remove();
							}
						}
					}
			}
		}
	}

	protected void copyFieldsFromParent() {
		System.arraycopy(parentFinder.r_sums, 0, r_sums, 0, rowHeaders.length);
		System.arraycopy(parentFinder.c_sums, 0, c_sums, 0, colHeaders.length);

		for (Map<Context, int[]> m : parentFinder.r_EctColSums) {
			Map<Context, int[]> m2 = new HashMap<>();
			for (Entry<Context, int[]> e : m.entrySet()) {
				m2.put(e.getKey(), new int[] { e.getValue()[0] });
			}
			r_EctColSums.add(m2);
		}
		for (Map<Context, int[]> m : parentFinder.c_EctColSums) {
			Map<Context, int[]> m2 = new HashMap<>();
			for (Entry<Context, int[]> e : m.entrySet()) {
				m2.put(e.getKey(), new int[] { e.getValue()[0] });
			}
			c_EctColSums.add(m2);
		}

		System.arraycopy(parentFinder.r_contributions, 0, r_contributions, 0,
				rowHeaders.length);
		System.arraycopy(parentFinder.c_contributions, 0, c_contributions, 0,
				colHeaders.length);

		for (Collection<Pattern[]> c : parentFinder.r_int_conflicts) {
			ArrayList<Pattern[]> c2 = new ArrayList<>(c);
			r_int_conflicts.add(c2);
		}
		for (Collection<Pattern[]> c : parentFinder.c_int_conflicts) {
			ArrayList<Pattern[]> c2 = new ArrayList<>(c);
			c_int_conflicts.add(c2);
		}
		for (Collection<Pair<Pattern, Pattern>> c : parentFinder.r_ext_conflicts) {
			ArrayList<Pair<Pattern, Pattern>> c2 = new ArrayList<>(c);
			r_ext_conflicts.add(c2);
		}
		for (Collection<Pair<Pattern, Pattern>> c : parentFinder.c_ext_conflicts) {
			ArrayList<Pair<Pattern, Pattern>> c2 = new ArrayList<>(c);
			c_ext_conflicts.add(c2);
		}

		validFields = true;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof BiClusterFinder) {
			BiClusterFinder f = (BiClusterFinder) obj;
			return bc.equals(f.bc);
		}
		return false;
	}

	@Override
	public int hashCode() {
		return bc.hashCode();
	}
}
