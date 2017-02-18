package aog.learn.bc;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import aog.Pattern;
import aog.relation.INoisyRelation;
import aog.relation.IRelation;
import aog.sample.Corpus;
import aog.sample.PatternInstance;
import aog.sample.Sample;
import aog.util.Matrix;
import aog.util.Pair;

/**
 * The symbol-symbol matrix used in the grammar learning algorithm. Each cell of
 * the matrix represents a pair of symbols and contains a set of pointers each
 * pointing to an appearance of the pair (at the first symbol of the pair).
 * 
 * @author Kewei Tu
 * 
 */
public class BigramMatrix extends
		Matrix<Pattern, Pattern, BigramMatrix.BigramCell> {

	// TODO In the case of noisy relation, add a relation search index to
	// quickly locate a range of relations

	public class BigramInstanceSet {
		public Set<Pair<PatternInstance, PatternInstance>> bgInstSet = new HashSet<>();

		public Map<Context, int[]> contextVector = new HashMap<>();

		/**
		 * The sets of pattern instances involved in each element of the bigram.
		 * Used to facilitate conflict discovery in biclustering.
		 */
		public Pair<Set<PatternInstance>, Set<PatternInstance>> ugInstSets = new Pair<Set<PatternInstance>, Set<PatternInstance>>(
				new HashSet<PatternInstance>(), new HashSet<PatternInstance>());

		public int size() {
			return bgInstSet.size();
		}

		public boolean isEmpty() {
			return bgInstSet.isEmpty();
		}

		public void add(Pair<PatternInstance, PatternInstance> i) {
			if (bgInstSet.add(i)) {
				Context c = CompositionalContext.getBigramContext(i.a,
						(CompositionalContext) pi2context.get(i.a), i.b,
						(CompositionalContext) pi2context.get(i.b));
				int[] count = contextVector.get(c);
				if (count == null) {
					count = new int[] { 0 };
					contextVector.put(c, count);
				}
				count[0]++;

				if (ugInstSets.a.contains(i.a) || ugInstSets.b.contains(i.b))
					System.err
							.println("[Warning] an element in bgm contains duplicate pattern instances!");
				ugInstSets.a.add(i.a);
				ugInstSets.b.add(i.b);
			}
		}

		public boolean remove(Pair<PatternInstance, PatternInstance> i) {
			boolean ret = bgInstSet.remove(i);
			if (ret) {
				Context c = CompositionalContext.getBigramContext(i.a,
						(CompositionalContext) pi2context.get(i.a), i.b,
						(CompositionalContext) pi2context.get(i.b));
				int[] count = contextVector.get(c);
				count[0]--;
				if (count[0] == 0)
					contextVector.remove(c);

				ugInstSets.a.remove(i.a);
				ugInstSets.b.remove(i.b);
			}
			return ret;
		}

		protected void removeContextCount(
				Pair<PatternInstance, PatternInstance> i) {
			if (bgInstSet.contains(i)) {
				Context c = CompositionalContext.getBigramContext(i.a,
						(CompositionalContext) pi2context.get(i.a), i.b,
						(CompositionalContext) pi2context.get(i.b));
				int[] count = contextVector.get(c);
				count[0]--;
				if (count[0] == 0)
					contextVector.remove(c);
			}
		}

		protected void addBackContextCount(
				Pair<PatternInstance, PatternInstance> i) {
			if (bgInstSet.contains(i)) {
				Context c = CompositionalContext.getBigramContext(i.a,
						(CompositionalContext) pi2context.get(i.a), i.b,
						(CompositionalContext) pi2context.get(i.b));
				int[] count = contextVector.get(c);
				if (count == null) {
					count = new int[] { 0 };
					contextVector.put(c, count);
				}
				count[0]++;
			}
		}
	}

	public class BigramCell {
		/**
		 * A map from a relation to the set of instances of this bigram that
		 * have this relation
		 */
		public Map<IRelation, BigramInstanceSet> rel2InstSet = new HashMap<>();

		// public HashMap<ExpressionContextMatrix.Context, Collection<Element>>
		// ecv = new HashMap<ExpressionContextMatrix.Context,
		// Collection<Element>>();
		//
		// public boolean isConsistent() {
		// int count = 0;
		// for (Map.Entry<Context, Collection<Element>> e : ecv.entrySet()) {
		// Collection<Element> c = e.getValue();
		// assert !c.isEmpty();
		// count += c.size();
		//
		// Context cont = e.getKey();
		// for (Element elem : c) {
		// Context cont2 = new Context(elem.prev, elem.next.next);
		// if (!cont.equals(cont2))
		// return false;
		// }
		// }
		// return count == num;
		// }
		//
		// /**
		// * Safe mode: when we can't guarantee that the pair's current context
		// is
		// * consistent with its record in this cell.
		// *
		// * @param app
		// * @return
		// */
		// public boolean removeSafeMode(Element app) {
		// for (Iterator<Collection<Element>> iter = ecv.values().iterator();
		// iter
		// .hasNext();) {
		// Collection<Element> c = (Collection<Element>) iter.next();
		// if (c.remove(app)) {
		// if (c.isEmpty())
		// iter.remove();
		// num--;
		// return true;
		// }
		// }
		// return false;
		// }

		public boolean remove(Pair<PatternInstance, PatternInstance> i) {
			IRelation r;
			try {
				r = (IRelation) relationType.newInstance();
				r = r.getRelation(i.a, i.b);
			} catch (InstantiationException | IllegalAccessException e) {
				e.printStackTrace();
				return false;
			}

			if (r == null)
				return false;

			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
				return false;
			} else {
				BigramInstanceSet instSet = rel2InstSet.get(r);
				if (instSet != null)
					if (instSet.remove(i)) {
						if (instSet.isEmpty()) {
							rel2InstSet.remove(r);
						}
						return true;
					}
				return false;
			}
		}

		// public void removeAll(Map<Context, Collection<Element>> ecv2) {
		// /*
		// * for (Map.Entry<Context, Collection<Element>> e :
		// * ecv2.entrySet()) { Collection<Element> c = ecv.get(e.getKey());
		// * if (c != null) if (c.removeAll(e.getValue())) if (c.isEmpty())
		// * ecv.remove(e.getKey()); } num = 0; for (Collection<Element> c :
		// * ecv.values()) { num += c.size(); }
		// */
		// /*
		// * A more efficient implementation under an assumption that is true
		// * in the current system.
		// */
		// for (Map.Entry<Context, Collection<Element>> e : ecv2.entrySet()) {
		// assert ecv.get(e.getKey()).equals(e.getValue());
		// // make sure of the assumption
		// ecv.remove(e.getKey());
		// num -= e.getValue().size();
		// }
		// }

		public boolean isEmpty() {
			return rel2InstSet.isEmpty();
		}

		public void add(Pair<PatternInstance, PatternInstance> i) {
			IRelation r;
			try {
				r = (IRelation) relationType.newInstance();
				r = r.getRelation(i.a, i.b);
			} catch (InstantiationException | IllegalAccessException e) {
				e.printStackTrace();
				return;
			}

			if (r == null)
				return;

			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
			} else {
				BigramInstanceSet instSet = rel2InstSet.get(r);
				if (instSet == null) {
					instSet = new BigramInstanceSet();
					rel2InstSet.put(r, instSet);
				}
				instSet.add(i);
			}
		}

		// public void addAll(Collection<Pair<PatternInstance, PatternInstance>>
		// is) {
		// instSet.addAll(is);
		// }

		public int size() {
			int size = 0;
			for (BigramInstanceSet instSet : rel2InstSet.values()) {
				size += instSet.size();
			}
			return size;
		}

		/**
		 * Return the bigram instances with the specified relation.
		 * 
		 * @param r
		 * @return
		 */
		public BigramInstanceSet getInstSetWithRelation(IRelation r) {
			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
				return null;
			} else {
				BigramInstanceSet s = rel2InstSet.get(r);
				return s;
			}
		}

		/**
		 * Return the number of bigram instances with the specified relation
		 * 
		 * @param r
		 * @return
		 */
		public int sizeWithRelation(IRelation r) {
			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
				return 0;
			} else {
				BigramInstanceSet s = rel2InstSet.get(r);
				if (s != null)
					return s.size();
				else
					return 0;
			}
		}

		@Override
		public String toString() {
			return Integer.toString(size());
		}

		protected void removeContextCount(
				Pair<PatternInstance, PatternInstance> i) {
			IRelation r;
			try {
				r = (IRelation) relationType.newInstance();
				r = r.getRelation(i.a, i.b);
			} catch (InstantiationException | IllegalAccessException e) {
				e.printStackTrace();
				return;
			}

			if (r == null)
				return;

			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
			} else {
				BigramInstanceSet instSet = rel2InstSet.get(r);
				if (instSet != null)
					instSet.removeContextCount(i);
			}
		}

		protected void addBackContextCount(
				Pair<PatternInstance, PatternInstance> i) {
			IRelation r;
			try {
				r = (IRelation) relationType.newInstance();
				r = r.getRelation(i.a, i.b);
			} catch (InstantiationException | IllegalAccessException e) {
				e.printStackTrace();
				return;
			}

			if (r == null)
				return;

			if (r instanceof INoisyRelation) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
			} else {
				BigramInstanceSet instSet = rel2InstSet.get(r);
				if (instSet != null)
					instSet.addBackContextCount(i);
			}
		}
	}

	public NeighborhoodJudge neighborhoodJudge = new DefaultNeighborhoodJudge();

	public void setNeighborhoodJudge(NeighborhoodJudge neighborhoodJudge) {
		this.neighborhoodJudge = neighborhoodJudge;
	}

	public Class<?> relationType;

	public void setRelationType(String relationType) {
		try {
			this.relationType = Class.forName(relationType);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	public Map<PatternInstance, Context> pi2context;

	public BigramMatrix(Map<PatternInstance, Context> pi2context) {
		super();
		this.pi2context = pi2context;
	}

	/**
	 * Construct a symbol-symbol matrix based on a corpus.
	 * 
	 * @param c
	 *            the corpus based on which this matrix is initialized.
	 */
	public void constructFromCorpus(Corpus c) {
		for (Sample s : c.samples) {
			for (int i = 0; i < s.elements.size(); i++) {
				PatternInstance a = s.elements.get(i);
				for (int j = i + 1; j < s.elements.size(); j++) {
					PatternInstance b = s.elements.get(j);
					if (neighborhoodJudge.isNeighbor(a, b, s)) {
						addAppearance(a, b);
						addAppearance(b, a);
					}
				}
			}
		}
	}

	public void addAppearance(PatternInstance a, PatternInstance b) {
		IRelation r;
		try {
			r = (IRelation) relationType.newInstance();
			r = r.getRelation(a, b);
		} catch (InstantiationException | IllegalAccessException e) {
			e.printStackTrace();
			return;
		}

		if (r == null)
			return;

		BigramCell c = getCell(a.type, b.type);
		if (c == null) {
			c = new BigramCell();
			setCell(a.type, b.type, c);
		}
		c.add(new Pair<PatternInstance, PatternInstance>(a, b));
	}

	/**
	 * @param a
	 * @param b
	 * @return if found and deleted
	 */
	public boolean delAppearance(PatternInstance a, PatternInstance b) {
		BigramCell cell = getCell(a.type, b.type);
		if (cell != null)
			if (cell.remove(new Pair<PatternInstance, PatternInstance>(a, b))) {
				if (cell.isEmpty())
					delCell(a.type, b.type);
				return true;
			}
		return false;
	}

	// /**
	// * Safe mode: when we can't guarantee that the pair's current context is
	// * consistent with its record in this ssm.
	// *
	// * @param app
	// */
	// public void delAppearanceSafeMode(Sentence.Element app) {
	// Symbol row = app.sym;
	// Symbol col = app.next.sym;
	// Map<Symbol, BigramCell> r = rows.get(row);
	// if (r != null) {
	// BigramCell apps = r.get(col);
	// if (apps != null)
	// if (apps.removeSafeMode(app))
	// if (apps.isEmpty())
	// delCell(row, col);
	// }
	// }

	// public void addRow(Symbol s, Map<Symbol, BigramCell> newRow) {
	// rows.put(s, newRow);
	// // rowHeaders.add((Symbol) s);
	// for (Map.Entry<Symbol, BigramCell> e : newRow.entrySet()) {
	// Symbol s2 = e.getKey();
	// if (!cols.containsKey(s2))
	// addCol(s2);
	// cols.get(s2).put(s, e.getValue());
	// cells.add(new Pair<Pattern,Pattern>(s, s2));
	// }
	// }
	//
	// public void addCol(Symbol s, Map<Symbol, BigramCell> newCol) {
	// cols.put(s, newCol);
	// // colHeaders.add((Symbol) s);
	// for (Map.Entry<Symbol, BigramCell> e : newCol.entrySet()) {
	// Symbol s2 = e.getKey();
	// if (!rows.containsKey(s2))
	// addRow(s2);
	// rows.get(s2).put(s, e.getValue());
	// cells.add(new Pair<Pattern,Pattern>(s2, s));
	// }
	// }
	//
	// @Override
	// public void addCell(Symbol r, Symbol c, BigramCell apps) {
	// super.addCell(r, c, apps);
	// cells.add(new Pair<Pattern,Pattern>(r, c));
	// }
	//
	// @Override
	// public void delCell(Symbol s1, Symbol s2) {
	// BigramCell apps = getCell(s1, s2);
	// if (apps != null)
	// cells.remove(new Pair<Pattern,Pattern>(s1, s2));
	// super.delCell(s1, s2);
	// }
	//
	// @Override
	// public void delCol(Symbol s) {
	// Map<Symbol, BigramCell> m = cols.remove(s);
	// if (m != null)
	// for (Symbol r : m.keySet()) {
	// Map<Symbol, BigramCell> m2 = rows.get(r);
	// m2.remove(s);
	// cells.remove(new Pair<Pattern,Pattern>(r, s));
	// if (m2.isEmpty())
	// rows.remove(r);
	// }
	// }
	//
	// @Override
	// public void delRow(Symbol s) {
	// Map<Symbol, BigramCell> m = rows.remove(s);
	// if (m != null)
	// for (Symbol c : m.keySet()) {
	// Map<Symbol, BigramCell> m2 = cols.get(c);
	// m2.remove(s);
	// cells.remove(new Pair<Pattern,Pattern>(s, c));
	// if (m2.isEmpty())
	// cols.remove(c);
	// }
	// }

	// /**
	// * For debugging. Check if the ssm is consistent, i.e. appearances stored
	// in
	// * each cell are consistent with the cell position.
	// *
	// * @return
	// */
	// protected boolean isConsistent() {
	// for (Map.Entry<Symbol, Map<Symbol, BigramCell>> e1 : rows
	// .entrySet()) {
	// for (Map.Entry<Symbol, BigramCell> e2 : e1.getValue()
	// .entrySet()) {
	// BigramCell c = e2.getValue();
	// assert !c.isEmpty();
	// assert c.isConsistent();
	// for (Collection<Element> d : c.ecv.values())
	// for (Sentence.Element app : d) {
	// if (app.next == null)
	// return false;
	// if (app.sym != e1.getKey()
	// || app.next.sym != e2.getKey())
	// return false;
	// }
	// }
	// }
	// return true;
	// }
	//
	// protected void printInconsistency() {
	// for (Map.Entry<Symbol, Map<Symbol, BigramCell>> e1 : rows
	// .entrySet()) {
	// for (Map.Entry<Symbol, BigramCell> e2 : e1.getValue()
	// .entrySet()) {
	// for (Collection<Element> d : e2.getValue().ecv.values())
	// for (Sentence.Element app : d) {
	// if (app.next == null)
	// System.out.println(e1.getKey() + "," + e2.getKey()
	// + ":" + app);
	// else if (app.sym != e1.getKey()
	// || app.next.sym != e2.getKey())
	// System.out.println(e1.getKey() + "," + e2.getKey()
	// + ":" + app);
	// }
	// }
	// }
	// }

	protected Set<Pattern> getCellsInRow(Pattern rh) {
		if (rows.containsKey(rh))
			return rows.get(rh).keySet();
		else
			return new HashSet<Pattern>();
	}

	protected Set<Pattern> getCellsInCol(Pattern ch) {
		if (cols.containsKey(ch))
			return cols.get(ch).keySet();
		else
			return new HashSet<Pattern>();
	}

	protected void removeContextCount(PatternInstance a, PatternInstance b) {
		BigramCell cell = getCell(a.type, b.type);
		if (cell != null)
			cell.removeContextCount(new Pair<PatternInstance, PatternInstance>(
					a, b));
	}

	protected void addBackContextCount(PatternInstance a, PatternInstance b) {
		BigramCell cell = getCell(a.type, b.type);
		if (cell != null)
			cell.addBackContextCount(new Pair<PatternInstance, PatternInstance>(
					a, b));
	}

	public BigramInstanceSet getCellWithRelation(Pattern a, Pattern b,
			IRelation r) {
		BigramCell cell = getCell(a, b);
		if (cell != null) {
			return cell.getInstSetWithRelation(r);
		}
		return null;
	}

	public int getCellSizeWithRelation(Pattern a, Pattern b, IRelation r) {
		BigramCell cell = getCell(a, b);
		if (cell != null) {
			return cell.sizeWithRelation(r);
		}
		return 0;
	}
}
