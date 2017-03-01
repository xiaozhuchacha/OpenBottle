package aog.learn.bc;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import aog.AndNode;
import aog.Grammar;
import aog.OrNode;
import aog.Pattern;
import aog.TerminalPattern;
import aog.learn.bc.BigramMatrix.BigramCell;
import aog.learn.bc.BigramMatrix.BigramInstanceSet;
import aog.relation.ICombiner;
import aog.relation.INegatableRelation;
import aog.relation.INoisyRelation;
import aog.relation.IParametricRelation;
import aog.relation.IRelation;
import aog.sample.Corpus;
import aog.sample.PatternInstance;
import aog.sample.Sample;
import aog.util.Matrix;
import aog.util.Pair;
import aog.util.Triple;
import aog.util.Util;

public class GrammarLearner {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// Set parameters
		CmdLineParser parser = new CmdLineParser(param);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			System.err.println(e.getMessage() + "\n");
			System.err.println("Full set of available options:");
			parser.printUsage(System.err);
			System.exit(1);
		}

		initialize();

		// Read corpus
		System.out.println("Reading the training corpus...");
		Corpus corpus = new Corpus();
		corpus.readFromFile(param.input);

		// Learning
		Grammar gram = learnGrammar(corpus);
		gram.saveToFile(param.output);
	}

	public static void initialize() throws InstantiationException,
			IllegalAccessException, ClassNotFoundException {
		Context.contextType = param.contextType;
		Context.range = param.contextRange;
		CompositionalContext.defaultCombiner = (ICombiner) Class.forName(
				param.combinerType).newInstance();

		BiClusterFinder.setParameters(param.alpha, param.dl1, param.dl2,
				param.dl3, param.mcbcMaxRatioToPrediction, param.beamSize,
				param.beamRelativeThreshold, param.allowConflict);
	}

	protected Grammar g;
	protected Corpus c;
	protected BigramMatrix bgm;

	protected double totalPosteriorGain;

	protected ArrayList<BiCluster> biClusterCandidates = new ArrayList<>();

	protected Map<AndNode, ArrayList<PatternInstance>> and2InstSet = new HashMap<>();

	protected Map<PatternInstance, Context> pi2context = new HashMap<>();

	protected boolean done = false;

	public GrammarLearner(Corpus corp) {
		c = corp;
		totalPosteriorGain = 0;

		for (Sample s : c.samples) {
			for (PatternInstance pi : s.elements) {
				Context cont = Context.newTerminalContext(pi);
				pi2context.put(pi, cont);
			}
		}

		g = new Grammar(c.terms);
		g.relationType = BiCluster.relationClass;
		try {
			g.combinerType = Class.forName(param.combinerType);
		} catch (ClassNotFoundException e1) {
			e1.printStackTrace();
			System.exit(1);
		}

		bgm = new BigramMatrix(pi2context);
		bgm.setRelationType(param.bgmRelationType);

		NeighborhoodJudge nj;
		try {
			Constructor<?> con = Class.forName(param.neighborhoodJudgeType)
					.getConstructor((Class<?>[]) null);
			nj = (NeighborhoodJudge) con.newInstance();
			if (nj instanceof DistNeighborhoodJudge) {
				((DistNeighborhoodJudge) nj)
						.setMaxDist(param.initMaxPatternDist);
			}
		} catch (SecurityException | ReflectiveOperationException
				| IllegalArgumentException e) {
			e.printStackTrace();
			System.out
					.println("Invalid neighborhood judge type. Using default...");
			nj = new DefaultNeighborhoodJudge();
		}
		bgm.setNeighborhoodJudge(nj);

		bgm.constructFromCorpus(c);
		System.out.println("Bigram matrix constructed: " + bgm.getStatistics());
	}

	/**
	 * A copy constructor
	 * 
	 * @param gl
	 */
	@SuppressWarnings("unchecked")
	public GrammarLearner(GrammarLearner gl) {
		// g
		g = new Grammar();
		g.relationType = gl.g.relationType;
		g.combinerType = gl.g.combinerType;
		assert gl.g.startSymbol == null;

		HashMap<Pattern, Pattern> patternMap = new HashMap<>();
		g.terms = new TerminalPattern[gl.g.terms.length];
		for (int i = 0; i < gl.g.terms.length; i++) {
			TerminalPattern t = gl.g.terms[i];
			TerminalPattern t2 = new TerminalPattern(t.obj);
			patternMap.put(t, t2);
			g.terms[i] = t2;
			// t2.id = t.id;
		}
		for (AndNode and : gl.g.andNodes) {
			AndNode and2 = new AndNode();
			patternMap.put(and, and2);
			g.andNodes.add(and2);
			// and2.id = and.id;
			and2.ccRelations = new IRelation[and.children.size()][and.children
					.size()];
			for (int i = 0; i < and.children.size(); i++) {
				for (int j = 0; j < and.children.size(); j++) {
					IRelation r = and.ccRelations[i][j];
					if (r != null)
						and2.ccRelations[i][j] = r.clone();
				}
			}
			and2.pcRelation = and.pcRelation.clone();
		}
		for (OrNode or : gl.g.orNodes) {
			OrNode or2 = new OrNode();
			patternMap.put(or, or2);
			g.orNodes.add(or2);
			// or2.id = or.id;
			or2.weights = (ArrayList<Double>) or.weights.clone();
		}

		for (TerminalPattern t : gl.g.terms) {
			TerminalPattern t2 = (TerminalPattern) patternMap.get(t);
			for (Pattern par : t.parents) {
				t2.parents.add(patternMap.get(par));
			}
		}
		for (AndNode and : gl.g.andNodes) {
			AndNode and2 = (AndNode) patternMap.get(and);
			for (Pattern chd : and.children) {
				and2.children.add(patternMap.get(chd));
			}
			for (Pattern par : and.parents) {
				and2.parents.add(patternMap.get(par));
			}
		}
		for (OrNode or : gl.g.orNodes) {
			OrNode or2 = (OrNode) patternMap.get(or);
			for (Pattern chd : or.children) {
				or2.children.add(patternMap.get(chd));
			}
			for (Pattern par : or.parents) {
				or2.parents.add(patternMap.get(par));
			}
		}

		// c
		c = new Corpus();
		c.terms = g.terms;
		c.samples = new Sample[gl.c.samples.length];
		HashMap<Sample, HashMap<PatternInstance, PatternInstance>> s2piMap = new HashMap<>();
		for (int i = 0; i < gl.c.samples.length; i++) {
			Sample s = gl.c.samples[i];
			Sample s2 = new Sample();
			HashMap<PatternInstance, PatternInstance> piMap = new HashMap<>();
			s2piMap.put(s, piMap);
			// copy fields other than contexts
			for (PatternInstance rpi : s.elements) {
				PatternInstance rpi2 = new PatternInstance();
				s2.elements.add(rpi2);
				ArrayList<Pair<PatternInstance, PatternInstance>> queue = new ArrayList<>();
				queue.add(new Pair<>(rpi, rpi2));
				for (int j = 0; j < queue.size(); j++) {
					rpi = queue.get(j).a;
					rpi2 = queue.get(j).b;
					piMap.put(rpi, rpi2);
					rpi2.type = patternMap.get(rpi.type);
					rpi2.parameters = (ArrayList<Double>) rpi.parameters
							.clone();
					rpi2.containingSample = s2;
					for (PatternInstance chd : rpi.children) {
						PatternInstance chd2 = new PatternInstance();
						rpi2.children.add(chd2);
						chd2.parent = rpi2;
						queue.add(new Pair<>(chd, chd2));
					}
				}
			}
			// copy contexts
			for (PatternInstance rpi : s.elements) {
				ArrayList<PatternInstance> queue = new ArrayList<>();
				queue.add(rpi);
				for (int j = 0; j < queue.size(); j++) {
					rpi = queue.get(j);
					PatternInstance rpi2 = piMap.get(rpi);
					Context cont2 = Context.newEmptyContext();
					pi2context.put(rpi2, cont2);
					cont2.centerParameters = rpi2.parameters;
					Context cont = gl.pi2context.get(rpi);
					for (PatternInstance e : cont.elements) {
						cont2.elements.add(piMap.get(e));
					}
					cont2.constructSignature();
					for (PatternInstance chd : rpi.children) {
						queue.add(chd);
					}
				}
			}
			c.samples[i] = s2;
		}

		// bgm
		bgm = new BigramMatrix(pi2context);
		bgm.relationType = gl.bgm.relationType;
		bgm.neighborhoodJudge = gl.bgm.neighborhoodJudge;
		for (Triple<Pattern, Pattern, BigramCell> triple : gl.bgm) {
			Pattern rh = triple.a;
			Pattern ch = triple.b;
			BigramCell cell = triple.c;
			BigramCell cell2 = bgm.new BigramCell();

			for (Entry<IRelation, BigramInstanceSet> e : cell.rel2InstSet
					.entrySet()) {
				IRelation r = e.getKey();
				IRelation r2;
				if (r instanceof IParametricRelation)
					r2 = ((IParametricRelation) r).clone();
				else
					r2 = r;

				BigramInstanceSet bis = e.getValue();
				BigramInstanceSet bis2 = bgm.new BigramInstanceSet();
				for (Pair<PatternInstance, PatternInstance> pair : bis.bgInstSet) {
					HashMap<PatternInstance, PatternInstance> piMap = s2piMap
							.get(pair.a.containingSample);
					PatternInstance pi2a = piMap.get(pair.a);
					PatternInstance pi2b = piMap.get(pair.b);
					bis2.add(new Pair<>(pi2a, pi2b));
				}

				cell2.rel2InstSet.put(r2, bis2);
			}

			bgm.setCell(patternMap.get(rh), patternMap.get(ch), cell2);
		}

		// totalPosteriorGain
		totalPosteriorGain = gl.totalPosteriorGain;

		// biClusterCandidates
		for (BiCluster bc : gl.biClusterCandidates) {
			BiCluster bc2 = new BiCluster();
			bc2.sum = bc.sum;
			bc2.posteriorGain = bc.posteriorGain;

			if (bc.relation instanceof IParametricRelation)
				bc2.relation = ((IParametricRelation) bc.relation).clone();
			else
				bc2.relation = bc.relation;

			for (Entry<Pattern, int[]> e : bc.rowSums.entrySet()) {
				bc2.rowSums.put(patternMap.get(e.getKey()),
						new int[] { e.getValue()[0] });
			}

			for (Entry<Pattern, int[]> e : bc.colSums.entrySet()) {
				bc2.colSums.put(patternMap.get(e.getKey()),
						new int[] { e.getValue()[0] });
			}

			// contextVector is not used later, so no copy

			biClusterCandidates.add(bc2);
		}

		// and2InstSet
		for (Entry<AndNode, ArrayList<PatternInstance>> e : gl.and2InstSet
				.entrySet()) {
			AndNode a2 = (AndNode) patternMap.get(e.getKey());
			ArrayList<PatternInstance> pis2 = new ArrayList<>();
			for (PatternInstance pi : e.getValue()) {
				PatternInstance pi2 = s2piMap.get(pi.containingSample).get(pi);
				pis2.add(pi2);
			}
			and2InstSet.put(a2, pis2);
		}
	}

	public static Grammar learnGrammar(Corpus c) {
		// Initialize BiCluster.relationClass
		try {
			BiCluster.relationClass = Class.forName(param.bcRelationType);
		} catch (ClassNotFoundException e1) {
			e1.printStackTrace();
			System.exit(1);
		}

		// The main loop
		GrammarLearner gl0 = new GrammarLearner(c);
		ArrayList<GrammarLearner> beam = new ArrayList<>();
		ArrayList<GrammarLearner> newBeam = new ArrayList<>();
		newBeam.add(gl0);
		boolean allDone = false;
		do {
			ArrayList<GrammarLearner> tmp = beam;
			beam = newBeam;
			tmp.clear();
			newBeam = tmp;

			for (int i = 0; i < beam.size(); i++) {
				GrammarLearner gl = beam.get(i);
				System.out.println("Learner #" + i + ":");
				if (gl.done) {
					System.out.println("I'm done...");
					addIntoBeam(newBeam, gl);
				} else {
					ArrayList<GrammarLearner> s = gl.learnOneStep();
					for (GrammarLearner gl2 : s) {
						addIntoBeam(newBeam, gl2);
					}
				}
			}
			System.out.println("Beam updated: " + newBeam);

			allDone = true;
			for (GrammarLearner gl : newBeam) {
				if (!gl.done) {
					allDone = false;
					break;
				}
			}
		} while (!allDone);

		// Post-processing: construct the start symbol
		System.out.println("No more learning. Post-processing...");
		double maxPosteriorGain = Double.NEGATIVE_INFINITY;
		Grammar bestGrammar = null;
		for (int i = 0; i < newBeam.size(); i++) {
			System.out.println("Learner #" + i + ":");
			GrammarLearner gl = newBeam.get(i);

			gl.postprocess();
			/*
			 * Tmp: use the following line instead for event parsing evaluation
			 * purpose
			 */
			// gl.g.startSymbol = gl.g.andNodes.get(0);

			System.out
					.println("Total posterior gain: " + gl.totalPosteriorGain);
			if (gl.totalPosteriorGain > maxPosteriorGain) {
				maxPosteriorGain = gl.totalPosteriorGain;
				bestGrammar = gl.g;
			}
		}

		return bestGrammar;
	}

	/**
	 * @param beam
	 *            The current beam in which items are sorted in descending order
	 * @param newItems
	 */
	protected static void addIntoBeam(ArrayList<GrammarLearner> beam,
			GrammarLearner newItem) {
		if (beam.contains(newItem))
			return;

		int i = Collections.binarySearch(beam, newItem,
				new Comparator<GrammarLearner>() {
					@Override
					public int compare(GrammarLearner o1, GrammarLearner o2) {
						return (int) Math.signum(o2.totalPosteriorGain
								- o1.totalPosteriorGain);
					}
				});
		i = i < 0 ? -i - 1 : i;
		if (i < param.beamSize) {
			beam.add(i, newItem);
			double threshold = beam.get(0).totalPosteriorGain
					- param.beamRelativeThreshold;
			for (int j = beam.size() - 1; j >= 1; j--) {
				if (j >= param.beamSize
						|| beam.get(j).totalPosteriorGain < threshold)
					beam.remove(j);
			}
		}
	}

	/**
	 * Learn the start rules.
	 */
	protected void postprocess() {
		HashMap<Pattern, double[]> sSet = new HashMap<>();
		ArrayList<AndNode> createdAndNodes = new ArrayList<>();
		int fullReductionCount = 0;
		SampleLoop: for (Sample s : c.samples) {
			Pattern top = null;
			if (s.elements.size() == 1) {
				fullReductionCount++;
				top = s.elements.get(0).type;
			} else {
				/*
				 * Sort s.elements to make the generated AND nodes easy to
				 * compare with each other
				 */
				ArrayList<PatternInstance> elements = new ArrayList<>(
						s.elements);
				Collections.sort(elements, new Comparator<PatternInstance>() {
					@Override
					public int compare(PatternInstance o1, PatternInstance o2) {
						if (o1.type.id != o2.type.id)
							return o1.type.id - o2.type.id;
						else {
							for (int i = 0; i < o1.parameters.size(); i++) {
								if (o1.parameters.get(i) != o2.parameters
										.get(i))
									return (int) Math.signum(o1.parameters
											.get(i) - o2.parameters.get(i));
							}
							return 0; // shouldn't happen
						}
					}
				});

				int n = elements.size();
				ArrayList<Pattern> children = new ArrayList<Pattern>(n);
				IRelation[][] ccRelations = new IRelation[n][n];
				for (int i = 0; i < elements.size(); i++) {
					PatternInstance pi = elements.get(i);
					children.add(pi.type);

					if (INoisyRelation.class
							.isAssignableFrom(BiCluster.relationClass)) {
						// TODO
						System.err
								.println("[Error] this part hasn't been implemented!");
						System.exit(1);
					} else {
						if (IParametricRelation.class
								.isAssignableFrom(bgm.relationType)) {
							/*
							 * We only need to store n-1 relations, since other
							 * relations can be computed from them
							 */
							IRelation r;
							try {
								r = (IRelation) bgm.relationType.newInstance();
								r = r.getRelation(pi, elements.get(0));
							} catch (InstantiationException
									| IllegalAccessException e) {
								e.printStackTrace();
								continue SampleLoop;
							}
							ccRelations[i][0] = r;
						} else {
							/*
							 * We have to check all n(n-1) relations
							 */
							for (int j = 0; j < children.size() - 1; j++) {
								PatternInstance pi2 = elements.get(j);

								IRelation r;
								try {
									r = (IRelation) bgm.relationType
											.newInstance();
									r = r.getRelation(pi, pi2);
								} catch (InstantiationException
										| IllegalAccessException e) {
									e.printStackTrace();
									continue SampleLoop;
								}
								ccRelations[i][j] = r;

								try {
									r = (IRelation) bgm.relationType
											.newInstance();
									r = r.getRelation(pi2, pi);
								} catch (InstantiationException
										| IllegalAccessException e) {
									e.printStackTrace();
									continue SampleLoop;
								}
								ccRelations[j][i] = r;
							}
						}
					}
				}

				// check duplication
				AndNode and = null;
				CreatedAndNodesLoop: for (AndNode and0 : createdAndNodes) {
					if (!and0.children.equals(children))
						continue;

					for (int i = 0; i < n; i++) {
						if (!Arrays.equals(and0.ccRelations[i], ccRelations[i]))
							continue CreatedAndNodesLoop;
					}

					and = and0;
					break;
				}

				if (and == null) {
					ICombiner pcRelation;
					try {
						pcRelation = (ICombiner) g.combinerType.newInstance();
					} catch (InstantiationException | IllegalAccessException e1) {
						e1.printStackTrace();
						continue SampleLoop;
					}

					and = new AndNode(children, ccRelations, pcRelation);
					createdAndNodes.add(and);
					g.andNodes.add(and);
				}
				top = and;
			}

			double[] v = sSet.get(top);
			if (v == null) {
				double[] tmp = { 1 };
				sSet.put(top, tmp);
			} else
				v[0]++;
		}

		System.out.println("Corpus full-reduction ratio: "
				+ ((double) fullReductionCount) / c.samples.length);

		ArrayList<Pattern> children = new ArrayList<>(sSet.size());
		ArrayList<Double> weights = new ArrayList<>(sSet.size());
		for (Map.Entry<Pattern, double[]> e : sSet.entrySet()) {
			children.add(e.getKey());
			weights.add(e.getValue()[0]);
		}
		g.startSymbol = new OrNode(children, weights);
		g.orNodes.add((OrNode) g.startSymbol);
	}

	protected ArrayList<GrammarLearner> learnOneStep() {

		if (param.debugLevel >= 2)
			for (Entry<PatternInstance, Context> e : pi2context.entrySet()) {
				PatternInstance pi = e.getKey();
				Context cont = e.getValue();
				ArrayList<PatternInstance> pips = new ArrayList<>();
				pips.add(pi);
				PatternInstance tmp = pi;
				while (tmp.parent != null) {
					tmp = tmp.parent;
					pips.add(tmp);
				}
				for (PatternInstance pi2 : cont.elements) {
					ArrayList<PatternInstance> pips2 = new ArrayList<>();
					pips2.add(pi2);
					tmp = pi2;
					while (tmp.parent != null) {
						tmp = tmp.parent;
						pips2.add(tmp);
					}

					// common ancestor
					int i = pips.size() - 1;
					int i2 = pips2.size() - 1;
					while (i > 0 && i2 > 0 && pips.get(i) == pips2.get(i2)) {
						i--;
						i2--;
					}

					Context cont2 = pi2context.get(pi2);
					assert cont2.elements.contains(pips.get(i));
				}
			}

		ArrayList<Integer> bestBcIndexes;
		if (param.growDistNeighborhood) {
			DistNeighborhoodJudge dnj = (DistNeighborhoodJudge) bgm.neighborhoodJudge;
			do {
				bestBcIndexes = bestBiClustering();
				if (bestBcIndexes.isEmpty()) {
					dnj.maxDist += param.stepSizeOfMaxPatternDist;
					System.out.println("Neighborhood distance set at "
							+ dnj.maxDist);
				} else
					break;
			} while (dnj.maxDist <= param.finalMaxPatternDist);
		} else
			bestBcIndexes = bestBiClustering();

		ArrayList<GrammarLearner> ret = new ArrayList<>();
		if (bestBcIndexes.isEmpty()) {
			done = true;
			ret.add(this);
		} else {
			for (int i = 0; i < bestBcIndexes.size(); i++) {
				GrammarLearner gl;
				if (i == bestBcIndexes.size() - 1)
					gl = this;
				else
					gl = new GrammarLearner(this);
				System.out
						.println("Fork this learner: #" + i + " (" + gl + ")");
				gl.learnFromBiCluster(bestBcIndexes.get(i));
				ret.add(gl);
			}
		}
		return ret;
	}

	/**
	 * Construct bgm from the corpus and do biclustering to update the grammar
	 * until no more valid bicluster can be found.
	 * 
	 * @param c
	 */
	protected void learnFromBiCluster(int indexInBcCandidates) {
		BiCluster bc = biClusterCandidates.remove(indexInBcCandidates);
		System.out.println("Learn from a bicluster: " + bc);

		totalPosteriorGain += bc.posteriorGain;

		// Construct the two OR nodes
		ArrayList<Pattern> children = new ArrayList<>();
		ArrayList<Double> weights = new ArrayList<>();
		bc.getRows(children, weights);
		OrNode o1 = new OrNode(children, weights);
		g.orNodes.add(o1);
		children = new ArrayList<>();
		weights = new ArrayList<>();
		bc.getCols(children, weights);
		OrNode o2 = new OrNode(children, weights);
		g.orNodes.add(o2);

		// Construct the AND node
		children = new ArrayList<Pattern>();
		children.add(o1);
		children.add(o2);
		IRelation[][] ccRelations = new IRelation[2][2];
		ccRelations[0][1] = bc.relation;
		ICombiner comb = null;
		try {
			comb = (ICombiner) g.combinerType.newInstance();
		} catch (InstantiationException | IllegalAccessException e1) {
			e1.printStackTrace();
			System.exit(1);
		}
		AndNode n = new AndNode(children, ccRelations, comb);
		g.andNodes.add(n);

		System.out.println("An AND-OR fragment added into the grammar.");

		// Do reduction
		Collection<Pair<PatternInstance, PatternInstance>> toReduce = new ArrayList<>();
		for (Pattern r : bc.rowSums.keySet()) {
			for (Pattern c : bc.colSums.keySet()) {
				BigramInstanceSet s = bgm
						.getCellWithRelation(r, c, bc.relation);
				if (s != null) {
					toReduce.addAll(s.bgInstSet);
				}
			}
		}
		doReduction(n, toReduce);

		// Find existing OR nodes that the new AND node should merge into
		/*
		 * TODO Enhancement: Currently we check OR nodes sequentially, and
		 * whenever an OR node leads to a posterior gain above the threshold, we
		 * will merge the AND node into it. A better way is to check all the
		 * candidate ORs and do merging in the order of the posterior gain, so
		 * if there is ambiguity (two ORs involve the same set of bigrams), we
		 * would choose the better one.
		 */
		HashSet<OrNode> candidates = new HashSet<OrNode>();
		for (Pattern p : bgm.getCellsInRow(n)) {
			for (Pattern pp : p.parents) {
				candidates.add((OrNode) pp);
			}
		}
		for (Pattern p : bgm.getCellsInCol(n)) {
			for (Pattern pp : p.parents) {
				candidates.add((OrNode) pp);
			}
		}
		for (OrNode or : candidates) {
			AndNode and = (AndNode) or.parents.get(0);
			int orIndex = and.children.indexOf(or);
			int or2Index = 1 - orIndex;
			OrNode or2 = (OrNode) and.children.get(or2Index);
			IRelation rel = and.ccRelations[0][1];

			// FIXME Check instance conflict. See allowConflict.

			// likelihood gain, part 1
			double sum = 0;
			int rsum = 0;
			for (int i = 0; i < or.children.size(); i++) {
				sum += or.weights.get(i);
				rsum += orIndex == 1 ? bgm.getCellSizeWithRelation(n,
						or.children.get(i), rel) : bgm.getCellSizeWithRelation(
						or.children.get(i), n, rel);
			}

			if (rsum == 0)
				continue;

			double lhg_1 = 0;
			lhg_1 += Util.xlogx(rsum);
			lhg_1 += 2 * Util.xlogx(sum);
			lhg_1 -= 2 * Util.xlogx(sum + rsum);
			for (int i = 0; i < or.children.size(); i++) {
				int ai = orIndex == 1 ? bgm.getCellSizeWithRelation(n,
						or.children.get(i), rel) : bgm.getCellSizeWithRelation(
						or.children.get(i), n, rel);
				if (ai != 0) {
					double ci = or.weights.get(i);
					lhg_1 -= Util.xlogx(ci);
					lhg_1 += Util.xlogx(ci + ai);
				} // else it's "+=0"
			}

			// likelihood gain, part 2
			HashMap<Context, int[]> ecv0 = new HashMap<>();
			ArrayList<PatternInstance> pis = and2InstSet.get(and);
			for (PatternInstance pi : pis) {
				Context cont = pi2context.get(pi);

				if (param.debugLevel >= 2) {
					ArrayList<PatternInstance> tmp = new ArrayList<>();
					PatternInstance pipa = pi.parent;
					while (pipa != null) {
						tmp.add(pipa);
						pipa = pipa.parent;
					}
					for (PatternInstance e : cont.elements) {
						assert e.parent == null || tmp.contains(e.parent);
					}
				}

				int[] count = ecv0.get(cont);
				if (count == null) {
					count = new int[] { 0 };
					ecv0.put(cont, count);
				}
				count[0]++;
			}

			double lhg_2 = 0;
			HashMap<Context, int[]> ecv = new HashMap<>();
			for (Pattern p : or.children) {
				BigramInstanceSet cwr = orIndex == 1 ? bgm.getCellWithRelation(
						n, p, rel) : bgm.getCellWithRelation(p, n, rel);
				if (cwr != null) {
					for (Map.Entry<Context, int[]> e : cwr.contextVector
							.entrySet()) {
						lhg_2 -= Util.xlogx(e.getValue()[0]);

						int[] colsum = ecv.get(e.getKey());
						if (colsum == null) {
							colsum = new int[] { 0 };
							ecv.put(e.getKey(), colsum);
						}
						colsum[0] += e.getValue()[0];
					}
				} // else it's "+=0"
			}

			for (Map.Entry<Context, int[]> e : ecv.entrySet()) {
				int newc = e.getValue()[0];
				int[] oldc = ecv0.get(e.getKey());
				if (oldc != null) {
					lhg_2 += Util.xlogx(oldc[0] + newc);
					lhg_2 -= Util.xlogx(oldc[0]);
				} else
					lhg_2 += Util.xlogx(newc);
			}

			// prior gain
			double priorGain = param.alpha
					* (-param.dl1 + param.dl2 * rsum + param.dl3
							* or.children.size());

			// Thresholding and merging
			double posteriorGain = lhg_1 + lhg_2 + priorGain;
			if (posteriorGain >= param.minPosteriorGain) {
				// 1. merge the AND to OR
				or2.children.add(n);
				n.parents.add(or2);
				for (int i = 0; i < or.children.size(); i++) {
					double wi = or.weights.get(i);
					double ai = orIndex == 1 ? bgm.getCellSizeWithRelation(n,
							or.children.get(i), rel)
							: bgm.getCellSizeWithRelation(or.children.get(i),
									n, rel);
					or.weights.set(i, wi + ai);
				}
				or2.weights.add((double) rsum);

				totalPosteriorGain += posteriorGain;

				System.out
						.println("The new AND is attached to an existing OR.");

				// 2. do reduction
				toReduce = new ArrayList<>();
				for (Pattern p : or.children) {
					BigramInstanceSet instSet = orIndex == 1 ? bgm
							.getCellWithRelation(n, p, rel) : bgm
							.getCellWithRelation(p, n, rel);
					if (instSet != null)
						toReduce.addAll(instSet.bgInstSet);
				}
				ArrayList<PatternInstance> newPIs = doReduction(and, toReduce);

				// 3. do further reductions
				for (int i = 0; i < newPIs.size(); i++) {
					PatternInstance pi = newPIs.get(i);
					/*
					 * check if pi has already been reduced when reducing some
					 * other pi
					 */
					if (pi.parent != null)
						continue;

					/*
					 * Note: Currently we check the parents of the pattern
					 * sequentially and do reduction once a valid reduction is
					 * found. A better way is to find all possible reductions
					 * and choose a best one.
					 */
					LoopPiParent: for (Pattern pattern : pi.type.parents) {
						OrNode orpa = (OrNode) pattern;
						AndNode andpa = (AndNode) orpa.parents.get(0);
						int indOrpa = andpa.children.indexOf(orpa);
						int indOrpa2 = 1 - indOrpa;
						OrNode orpa2 = (OrNode) andpa.children.get(indOrpa2);
						IRelation relpa = andpa.ccRelations[0][1];

						for (PatternInstance pi2 : pi.containingSample.elements) {
							if (orpa2.children.contains(pi2.type)
									&& (indOrpa == 0 ? relpa.hasRelation(pi,
											pi2) : relpa.hasRelation(pi2, pi))) {
								// update the weights of ORs
								int ind = orpa.children.indexOf(pi.type);
								double w = orpa.weights.get(ind);
								orpa.weights.set(ind, w + 1);
								ind = orpa2.children.indexOf(pi2.type);
								w = orpa2.weights.get(ind);
								orpa2.weights.set(ind, w + 1);

								// update the total posterior gain
								/*
								 * Here we ignore the likelihood gain as
								 * explained in the notes. TODO compute it
								 */
								priorGain = param.alpha * param.dl2;
								totalPosteriorGain += priorGain;

								// do reduction
								ArrayList<Pair<PatternInstance, PatternInstance>> pair = new ArrayList<>();
								if (indOrpa == 1)
									pair.add(new Pair<PatternInstance, PatternInstance>(
											pi2, pi));
								else
									pair.add(new Pair<PatternInstance, PatternInstance>(
											pi, pi2));
								ArrayList<PatternInstance> reducedPI = doReduction(
										andpa, pair);

								// store the reduced pi for further
								// reduction
								newPIs.addAll(reducedPI);

								break LoopPiParent;
							}
						}
					}
				}
			}
		}

		System.out.println("Bigram matrix updated: " + bgm.getStatistics());
	}

	protected ArrayList<PatternInstance> doReduction(AndNode and,
			Collection<Pair<PatternInstance, PatternInstance>> toReduce) {
		// group pi in toReduce by their sample
		Map<Sample, Collection<Pair<PatternInstance, PatternInstance>>> sample2ToReduce = new HashMap<>();
		for (Pair<PatternInstance, PatternInstance> pair : toReduce) {
			Sample s = pair.a.containingSample;
			Collection<Pair<PatternInstance, PatternInstance>> c = sample2ToReduce
					.get(s);
			if (c == null) {
				c = new ArrayList<Pair<PatternInstance, PatternInstance>>();
				sample2ToReduce.put(s, c);
			}
			c.add(pair);
		}

		/*
		 * 1. remove the pattern instances to be reduced from bgm; create the
		 * new pattern instances; collect the instances whose context is
		 * affected
		 */
		Map<Sample, Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>>> removedAffectedNewPIs = new HashMap<>();
		ArrayList<PatternInstance> newPISet = new ArrayList<>();
		for (Entry<Sample, Collection<Pair<PatternInstance, PatternInstance>>> e : sample2ToReduce
				.entrySet()) {
			Sample s = e.getKey();
			Collection<Pair<PatternInstance, PatternInstance>> c = e.getValue();

			// save 1) removed pi; 2) pi whose context is affected;
			// 3) new pi
			Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>> sets = new Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>>(
					new HashSet<PatternInstance>(),
					new HashSet<PatternInstance>(),
					new ArrayList<PatternInstance>());

			// create the new instance; remove the reduced instances from bgm
			// but not sample (so no context is changed as old contexts are
			// needed in later bgm operations)
			for (Pair<PatternInstance, PatternInstance> pair : c) {
				PatternInstance pia = pair.a;
				PatternInstance pib = pair.b;

				/*
				 * One or both instances in this pair may have already been
				 * reduced. In that case, we ignore this pair. Note that for
				 * such conflicting reductions, we do not check which is better
				 * and simply reduce the one seen first.
				 */
				if (sets.a.contains(pia) || sets.a.contains(pib)) {
					/*
					 * Adjust the weights in the OR nodes.
					 */
					OrNode o1 = (OrNode) and.children.get(0);
					int ind = o1.children.indexOf(pia.type);
					double w = o1.weights.get(ind);
					if (w == 1) {
						o1.children.remove(ind);
						o1.weights.remove(ind);
						pia.type.parents.remove(o1);
					} else {
						o1.weights.set(ind, w - 1);
					}

					OrNode o2 = (OrNode) and.children.get(1);
					ind = o2.children.indexOf(pib.type);
					w = o2.weights.get(ind);
					if (w == 1) {
						o2.children.remove(ind);
						o2.weights.remove(ind);
						pib.type.parents.remove(o2);
					} else {
						o2.weights.set(ind, w - 1);
					}

					continue;
				}

				ArrayList<ArrayList<Double>> paramOfChildren = new ArrayList<>();
				paramOfChildren.add(pia.parameters);
				paramOfChildren.add(pib.parameters);
				ArrayList<Double> param = and.pcRelation
						.getParentFromChildren(paramOfChildren);
				PatternInstance newPI = new PatternInstance(and, param,
						pia.containingSample,
						new PatternInstance[] { pia, pib });
				newPISet.add(newPI);

				sets.a.add(pia);
				sets.a.add(pib);
				Context cont = pi2context.get(pia);
				sets.b.addAll(cont.elements);
				cont = pi2context.get(pib);
				sets.b.addAll(cont.elements);
				sets.c.add(newPI);

				PatternInstance[] tmp = new PatternInstance[] { pia, pib };
				for (PatternInstance pi : tmp)
					for (PatternInstance pi2 : s.elements) {
						if (pi2 != pi
								&& bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
							bgm.delAppearance(pi, pi2);
							bgm.delAppearance(pi2, pi);
						}
					}
			}

			removedAffectedNewPIs.put(s, sets);
		}

		if (and2InstSet.containsKey(and)) {
			ArrayList<PatternInstance> oldPISet = and2InstSet.get(and);
			oldPISet.addAll(newPISet);
		} else
			and2InstSet.put(and, newPISet);

		/*
		 * 2. update the contexts of instances other than the new instances;
		 * update accordingly the EC matrices in bgm; update the samples
		 */
		for (Entry<Sample, Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>>> e : removedAffectedNewPIs
				.entrySet()) {
			Sample s = e.getKey();
			Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>> sets = e
					.getValue();
			Set<PatternInstance> removed = sets.a;
			Set<PatternInstance> affected = sets.b;
			ArrayList<PatternInstance> newPIs = sets.c;

			ArrayList<PatternInstance> unaffected = new ArrayList<>(s.elements);
			unaffected.removeAll(affected);
			unaffected.removeAll(newPIs);

			// for affected\removed pi, remove its entry from the ecv in bgm
			ArrayList<PatternInstance> affDiffRem = new ArrayList<>(affected);
			affDiffRem.removeAll(removed);
			for (int i = 0; i < affDiffRem.size(); i++) {
				PatternInstance pi = affDiffRem.get(i);

				for (PatternInstance pi2 : unaffected) {
					if (bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
						bgm.removeContextCount(pi, pi2);
						bgm.removeContextCount(pi2, pi);
					}
				}

				for (int j = i + 1; j < affDiffRem.size(); j++) {
					PatternInstance pi2 = affDiffRem.get(j);
					if (bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
						bgm.removeContextCount(pi, pi2);
						bgm.removeContextCount(pi2, pi);
					}
				}
			}

			// add the new instance into and remove the reduced instances from
			// sample
			s.elements.removeAll(sets.a);
			s.elements.addAll(sets.c);

			// update context for affected pi (including reduced pi)
			for (PatternInstance pi : affected) {
				for (PatternInstance newPI : newPIs) {
					PatternInstance pia = newPI.children.get(0);
					PatternInstance pib = newPI.children.get(1);
					if (pi == pia || pi == pib)
						continue;

					// traverse all the pis reduced to this pi
					ArrayList<PatternInstance> piList = new ArrayList<>();
					piList.add(pi);
					for (int i = 0; i < piList.size(); i++) {
						PatternInstance pii = piList.get(i);
						CompositionalContext cont = (CompositionalContext) pi2context
								.get(pii);
						cont.updateWithBigramReduction(pia, pib, newPI);

						piList.addAll(pii.children);
					}
				}
			}

			// for affected\removed pi, add its entry back to ecv
			for (int i = 0; i < affDiffRem.size(); i++) {
				PatternInstance pi = affDiffRem.get(i);

				for (PatternInstance pi2 : unaffected) {
					if (bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
						bgm.addBackContextCount(pi, pi2);
						bgm.addBackContextCount(pi2, pi);
					}
				}

				for (int j = i + 1; j < affDiffRem.size(); j++) {
					PatternInstance pi2 = affDiffRem.get(j);
					if (bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
						bgm.addBackContextCount(pi, pi2);
						bgm.addBackContextCount(pi2, pi);
					}
				}
			}
		}

		/*
		 * 3. construct the context of the new pattern instances created from
		 * bc; add the new pattern instances to the bigram matrix
		 */
		for (Triple<Set<PatternInstance>, Set<PatternInstance>, ArrayList<PatternInstance>> t : removedAffectedNewPIs
				.values()) {
			ArrayList<PatternInstance> newPIs = t.c;
			for (Iterator<PatternInstance> iter = newPIs.iterator(); iter
					.hasNext();) {
				PatternInstance pi = iter.next();
				PatternInstance pia = pi.children.get(0);
				PatternInstance pib = pi.children.get(1);
				pi2context.put(pi, CompositionalContext.getBigramContext(pia,
						(CompositionalContext) pi2context.get(pia), pib,
						(CompositionalContext) pi2context.get(pib)));

				Sample s = pi.containingSample;
				for (PatternInstance pi2 : s.elements) {
					if (!newPIs.contains(pi2)
							&& bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
						bgm.addAppearance(pi, pi2);
						bgm.addAppearance(pi2, pi);
					}
				}

				iter.remove();
			}
		}

		return newPISet;
	}

	/**
	 * Do multiple bi-clustering on bgm and return the bi-cluster with the
	 * highest posterior gain.
	 * 
	 * @return the indexes of the best bi-clusters in
	 *         {@link #biClusterCandidates} , or an empty list if no bi-cluster
	 *         is found.
	 */
	protected ArrayList<Integer> bestBiClustering() {
		if (bgm.isEmpty())
			return new ArrayList<>();
		BiClusterFinder.setBigramMatrix(bgm);

		double bestGain = Double.NEGATIVE_INFINITY;
		BiCluster bestBC = null;
		HashSet<BiCluster> newCandidates = new HashSet<BiCluster>();

		// Initialize single-cell seeds
		Matrix<Pattern, Pattern, Set<IRelation>> seeds = new Matrix<>();
		int nSeeds = 0;
		for (Triple<Pattern, Pattern, BigramCell> cell : bgm) {
			Set<IRelation> s = cell.c.rel2InstSet.keySet();
			seeds.setCell(cell.a, cell.b, new HashSet<>(s));
			nSeeds += s.size();
		}
		int nCheckedSeed = 0;

		// Check and update existing candidates
		for (BiCluster bc : biClusterCandidates) {
			ArrayList<Pattern> rows = new ArrayList<>(bc.rowSums.keySet());
			rows.retainAll(bgm.getRowHeaders());
			if (!rows.isEmpty()) {
				ArrayList<Pattern> cols = new ArrayList<>(bc.colSums.keySet());
				cols.retainAll(bgm.getColHeaders());
				if (!cols.isEmpty()) {
					IRelation rel;
					if (bc.relation instanceof IParametricRelation)
						rel = ((IParametricRelation) bc.relation).clone();
					else
						rel = bc.relation;

					Collection<BiCluster> bcs = BiClusterFinder.biClustering(
							rows, cols, rel);
					for (BiCluster bc2 : bcs) {
						if (newCandidates.add(bc2)) {
							if (bestGain < bc2.posteriorGain) {
								bestGain = bc2.posteriorGain;
								bestBC = bc2;
							}

							// remove containing seeds
							for (Pattern r : bc2.rowSums.keySet()) {
								for (Pattern c : bc2.colSums.keySet()) {
									if (INoisyRelation.class
											.isAssignableFrom(BiCluster.relationClass)) {
										// TODO
										System.err
												.println("[Error] this part hasn't been implemented!");
										System.exit(1);
									} else {
										Set<IRelation> s = seeds.getCell(r, c);
										rel = bc2.relation;
										if (s != null && s.remove(rel)) {
											nCheckedSeed++;
											if (s.isEmpty())
												seeds.delCell(r, c);
											// transpose
											if (rel instanceof INegatableRelation) {
												rel = ((INegatableRelation) rel)
														.negate();
												s = seeds.getCell(c, r);
												if (s != null && s.remove(rel)) {
													nCheckedSeed++;
													if (s.isEmpty())
														seeds.delCell(c, r);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		// Find new candidates
		ArrayList<Triple<Pattern, Pattern, Set<IRelation>>> seedList = new ArrayList<>();
		for (Triple<Pattern, Pattern, Set<IRelation>> t : seeds) {
			seedList.add(t);
		}
		while (nCheckedSeed < nSeeds * param.ratioOfSeedsForBiClustering) {
			// seeding
			int ind = (int) (Util.random() * seedList.size());
			Triple<Pattern, Pattern, Set<IRelation>> seed = seedList.get(ind);
			if (ind == seedList.size() - 1)
				seedList.remove(ind);
			else
				seedList.set(ind, seedList.remove(seedList.size() - 1));
			if (seed.c.isEmpty())
				continue;
			Pattern r0 = seed.a, c0 = seed.b;
			ArrayList<IRelation> rels = new ArrayList<>(seed.c);

			int checkedSize = seed.c.size();
			nCheckedSeed += checkedSize;
			seed.c.clear();
			if (INegatableRelation.class.isAssignableFrom(bgm.relationType)) {
				Set<IRelation> s = seeds.getCell(c0, r0);
				if (s != null) {
					nCheckedSeed += s.size();
					s.clear();
				}
			}

			// biClustering
			ArrayList<Pattern> rows = new ArrayList<>();
			rows.add(r0);
			ArrayList<Pattern> cols = new ArrayList<>();
			cols.add(c0);
			if (INoisyRelation.class.isAssignableFrom(BiCluster.relationClass)) {
				// TODO
				System.err
						.println("[Error] this part hasn't been implemented!");
				System.exit(1);
			} else {
				if (!param.bgmRelationType.equals(param.bcRelationType)) {
					System.err
							.println("[Error] bgmRelationType should be set to the same as bcRelationType when both are not INoisyRelation!");
					System.exit(1);
				}

				/*
				 * Here we check all the relations, which may be problematic if
				 * the number of relations is large.
				 */
				for (IRelation rel : rels) {
					Collection<BiCluster> bcs = BiClusterFinder.biClustering(
							rows, cols, rel);
					for (BiCluster bc : bcs) {
						if (newCandidates.add(bc)) {
							if (bestGain < bc.posteriorGain) {
								bestGain = bc.posteriorGain;
								bestBC = bc;
							}

							// remove containing seeds
							for (Pattern r : bc.rowSums.keySet()) {
								for (Pattern c : bc.colSums.keySet()) {
									if (INoisyRelation.class
											.isAssignableFrom(BiCluster.relationClass)) {
										// TODO
										System.err
												.println("[Error] this part hasn't been implemented!");
										System.exit(1);
									} else {
										Set<IRelation> s = seeds.getCell(r, c);
										rel = bc.relation;
										if (s != null && s.remove(rel)) {
											nCheckedSeed++;
											// transpose
											if (rel instanceof INegatableRelation) {
												s = seeds.getCell(c, r);
												rel = ((INegatableRelation) rel)
														.negate();
												if (s != null && s.remove(rel)) {
													nCheckedSeed++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		if (bestGain < param.minPosteriorGain)
			return new ArrayList<>(0);

		// construct the beam
		double threshold = Math.max(param.minPosteriorGain, bestGain
				- param.beamRelativeThreshold);
		ArrayList<BiCluster> beam = new ArrayList<>();
		for (BiCluster bc : newCandidates) {
			if (bc.posteriorGain < threshold)
				continue;
			if (param.beamConflictingBcOnly && bc != bestBC
					&& !isConflicting(bc, bestBC))
				continue;

			int i = Collections.binarySearch(beam, bc,
					new Comparator<BiCluster>() {
						@Override
						public int compare(BiCluster o1, BiCluster o2) {
							return (int) Math.signum(o2.posteriorGain
									- o1.posteriorGain);
						}
					});
			i = i < 0 ? -i - 1 : i;
			if (i < param.beamSize) {
				beam.add(i, bc);
				if (beam.size() > param.beamSize)
					beam.remove(param.beamSize);
			}
		}

		biClusterCandidates.clear();
		biClusterCandidates.addAll(newCandidates);

		// sort the candidates
		// Collections.sort(biClusterCandidates, new Comparator<BiCluster>() {
		// @Override
		// public int compare(BiCluster o1, BiCluster o2) {
		// return (int) Math.signum(o2.posteriorGain - o1.posteriorGain);
		// }
		// });

		// Return the best one
		ArrayList<Integer> ret = new ArrayList<>(param.beamSize);
		for (BiCluster bc : beam) {
			ret.add(biClusterCandidates.indexOf(bc));
		}
		return ret;
	}

	protected boolean isConflicting(BiCluster bc1, BiCluster bc2) {
		if (bc1.conflicts(bc2))
			return true;

		for (Pattern r1 : bc1.rowSums.keySet()) {
			if (bc2.rowSums.containsKey(r1)) {
				for (Pattern c1 : bc1.colSums.keySet()) {
					BigramInstanceSet bis1 = bgm.getCellWithRelation(r1, c1,
							bc1.relation);
					if (bis1 == null)
						continue;
					Set<PatternInstance> set1 = bis1.ugInstSets.a;
					for (Pattern c2 : bc2.colSums.keySet()) {
						BigramInstanceSet bis2 = bgm.getCellWithRelation(r1,
								c2, bc2.relation);
						if (bis2 == null)
							continue;
						Set<PatternInstance> set2 = bis2.ugInstSets.a;
						if (!Collections.disjoint(set1, set2))
							return true;
					}
				}
			}

			if (bc2.colSums.containsKey(r1)) {
				for (Pattern c1 : bc1.colSums.keySet()) {
					BigramInstanceSet bis1 = bgm.getCellWithRelation(r1, c1,
							bc1.relation);
					if (bis1 == null)
						continue;
					Set<PatternInstance> set1 = bis1.ugInstSets.a;
					for (Pattern r2 : bc2.rowSums.keySet()) {
						BigramInstanceSet bis2 = bgm.getCellWithRelation(r2,
								r1, bc2.relation);
						if (bis2 == null)
							continue;
						Set<PatternInstance> set2 = bis2.ugInstSets.b;
						if (!Collections.disjoint(set1, set2))
							return true;
					}
				}
			}
		}

		for (Pattern c1 : bc1.colSums.keySet()) {
			if (bc2.rowSums.containsKey(c1)) {
				for (Pattern r1 : bc1.rowSums.keySet()) {
					BigramInstanceSet bis1 = bgm.getCellWithRelation(r1, c1,
							bc1.relation);
					if (bis1 == null)
						continue;
					Set<PatternInstance> set1 = bis1.ugInstSets.b;
					for (Pattern c2 : bc2.colSums.keySet()) {
						BigramInstanceSet bis2 = bgm.getCellWithRelation(c1,
								c2, bc2.relation);
						if (bis2 == null)
							continue;
						Set<PatternInstance> set2 = bis2.ugInstSets.a;
						if (!Collections.disjoint(set1, set2))
							return true;
					}
				}
			}

			if (bc2.colSums.containsKey(c1)) {
				for (Pattern r1 : bc1.rowSums.keySet()) {
					BigramInstanceSet bis1 = bgm.getCellWithRelation(r1, c1,
							bc1.relation);
					if (bis1 == null)
						continue;
					Set<PatternInstance> set1 = bis1.ugInstSets.b;
					for (Pattern r2 : bc2.rowSums.keySet()) {
						BigramInstanceSet bis2 = bgm.getCellWithRelation(r2,
								c1, bc2.relation);
						if (bis2 == null)
							continue;
						Set<PatternInstance> set2 = bis2.ugInstSets.b;
						if (!Collections.disjoint(set1, set2))
							return true;
					}
				}
			}
		}

		return false;
	}

	protected void delPatternInstance(PatternInstance pi) {
		Sample s = pi.containingSample;

		for (PatternInstance pi2 : s.elements) {
			if (pi2 != pi && bgm.neighborhoodJudge.isNeighbor(pi, pi2, s)) {
				bgm.delAppearance(pi, pi2);
				bgm.delAppearance(pi2, pi);
			}
		}

		s.elements.remove(pi);
	}

	@Override
	public int hashCode() {
		// TODO Auto-generated method stub
		return super.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		/*
		 * Here we simply compare the two partial grammars and do not compare
		 * the partially reduced corpora.
		 */
		/*
		 * Checking whether the two grammars are equivalent is probably NP-hard.
		 * So we assume the grammar is a DAG, i.e., it contains no recursive
		 * rule; we (quite safely) assume the grammar contains no duplicate
		 * terminals (in terms of the contained object) and AND-OR groups (in
		 * terms of the relations, children and weights); and we make other
		 * simplifications as noted in the code.
		 */
		if (!(obj instanceof GrammarLearner))
			return false;

		GrammarLearner gl2 = (GrammarLearner) obj;
		Grammar g2 = gl2.g;

		if (g.relationType != g2.relationType
				|| g.combinerType != g2.combinerType
				|| g.terms.length != g2.terms.length
				|| g.andNodes.size() != g2.andNodes.size()
				|| g.orNodes.size() != g2.orNodes.size())
			return false;

		HashMap<Pattern, Pattern> patternMap = new HashMap<>();

		// compare terminals
		for (TerminalPattern t : g.terms) {
			boolean matched = false;
			for (TerminalPattern t2 : g2.terms) {
				if (t.hasEqualObject(t2)
						&& t.parents.size() == t2.parents.size()) {
					matched = true;
					patternMap.put(t, t2);
					break;
				}
			}
			if (!matched)
				return false;
		}

		// compare AND-OR groups
		HashMap<Pattern, int[]> childCount = new HashMap<>();
		for (AndNode and : g.andNodes) {
			childCount.put(and, new int[] { and.children.size() });
		}
		for (OrNode or : g.orNodes) {
			childCount.put(or, new int[] { or.children.size() });
		}
		LinkedList<AndNode> queue = new LinkedList<>();
		for (TerminalPattern t : g.terms) {
			for (Pattern or : t.parents) {
				int[] count = childCount.get(or);
				count[0]--;
				if (count[0] == 0) {
					Pattern and = or.parents.get(0);
					count = childCount.get(and);
					count[0]--;
					if (count[0] == 0) {
						queue.add((AndNode) and);
					}
				}
			}
		}

		HashMap<Pattern, int[]> childCount2 = new HashMap<>();
		for (AndNode and : g2.andNodes) {
			childCount2.put(and, new int[] { and.children.size() });
		}
		for (OrNode or : g2.orNodes) {
			childCount2.put(or, new int[] { or.children.size() });
		}
		LinkedList<AndNode> queue2 = new LinkedList<>();
		for (TerminalPattern t : g2.terms) {
			for (Pattern or : t.parents) {
				int[] count = childCount2.get(or);
				count[0]--;
				if (count[0] == 0) {
					Pattern and = or.parents.get(0);
					count = childCount2.get(and);
					count[0]--;
					if (count[0] == 0) {
						queue2.add((AndNode) and);
					}
				}
			}
		}

		while (!queue.isEmpty()) {
			AndNode a = queue.pop();
			boolean matched = false;
			for (Iterator<AndNode> iterator = queue2.iterator(); iterator
					.hasNext();) {
				AndNode a2 = iterator.next();
				if (a.children.size() != a2.children.size()
						|| a.parents.size() != a2.parents.size())
					continue;

				// compare children
				boolean transposed;
				OrNode o1 = (OrNode) a.children.get(0);
				OrNode o2 = (OrNode) a.children.get(1);
				OrNode o21 = (OrNode) a2.children.get(0);
				OrNode o22 = (OrNode) a2.children.get(1);
				if (equalOrNode(o1, o21, patternMap)
						&& equalOrNode(o2, o22, patternMap))
					transposed = false;
				else if (equalOrNode(o1, o22, patternMap)
						&& equalOrNode(o2, o21, patternMap))
					transposed = true;
				else
					continue;

				// compare relations
				IRelation r = a.ccRelations[0][1];
				IRelation r2 = a2.ccRelations[0][1];
				if (transposed) {
					if (r instanceof INegatableRelation) {
						if (r instanceof INoisyRelation) {
							INoisyRelation nr = (INoisyRelation) r;
							INoisyRelation nr2 = (INoisyRelation) r2;
							if (!nr.approxEquals((INoisyRelation) nr2.negate()))
								continue;
						} else {
							INegatableRelation pr = (INegatableRelation) r;
							INegatableRelation pr2 = (INegatableRelation) r2;
							if (!pr.equals(pr2.negate()))
								continue;
						}
					} else
						continue;
				} else {
					if (r instanceof INoisyRelation) {
						INoisyRelation nr = (INoisyRelation) r;
						INoisyRelation nr2 = (INoisyRelation) r2;
						if (!nr.approxEquals(nr2))
							continue;
					} else {
						if (!r.equals(r2))
							continue;
					}
				}

				if (!a.pcRelation.equals(a2.pcRelation))
					continue;

				// matched
				matched = true;
				iterator.remove();
				patternMap.put(a, a2);
				for (Pattern or : a.parents) {
					int[] count = childCount.get(or);
					count[0]--;
					if (count[0] == 0) {
						Pattern and = or.parents.get(0);
						count = childCount.get(and);
						count[0]--;
						if (count[0] == 0) {
							queue.add((AndNode) and);
						}
					}
				}
				for (Pattern or : a2.parents) {
					int[] count = childCount2.get(or);
					count[0]--;
					if (count[0] == 0) {
						Pattern and = or.parents.get(0);
						count = childCount2.get(and);
						count[0]--;
						if (count[0] == 0) {
							queue2.add((AndNode) and);
						}
					}
				}
				break;
			}

			if (!matched)
				return false;
		}

		return true;
	}

	protected boolean equalOrNode(OrNode o1, OrNode o2,
			HashMap<Pattern, Pattern> patternMap) {
		if (o1.children.size() != o2.children.size()
				|| o1.parents.size() != o2.parents.size())
			return false;

		for (int i = 0; i < o1.children.size(); i++) {
			Pattern p1 = o1.children.get(i);
			Pattern p2 = patternMap.get(p1);
			int i2 = o2.children.indexOf(p2);
			if (i2 == -1)
				return false;
			else {
				if (!o1.weights.get(i).equals(o2.weights.get(i2)))
					return false;
			}
		}

		return true;
	}

	public static class Parameters {
		/**
		 * The input corpus file name
		 */
		@Option(name = "-input", usage = "The input corpus file name", required = true)
		public String input;

		/**
		 * The output grammar file name
		 */
		@Option(name = "-output", usage = "The output grammar file name", required = true)
		public String output;

		/**
		 * The level of outputting debug info and doing validation. <br>
		 * <li>0: no debugging <li>1: output debugging info <li>2: perform
		 * validations
		 */
		@Option(name = "-debugLevel", usage = "The level of outputting debug info and doing validation")
		public int debugLevel = 0;

		/*------------------------------------------------*/
		/*------------- Algorithm Parameters -------------*/
		/*------------------------------------------------*/

		/**
		 * The type of context. Must be a compositional context.
		 */
		@Option(name = "-contextType", usage = "The full class name of the context type")
		public String contextType = "aog.learn.bc.DefaultContext";
		/**
		 * The range around a pattern instance that is considered in computing
		 * its distance based context
		 */
		@Option(name = "-contextRange", usage = "The context range")
		public double contextRange = 5;

		/**
		 * The type of relation used in the bigram matrix
		 */
		@Option(name = "-bgmRelationType", usage = "The full class name of the relation used in the bigram matrix")
		public String bgmRelationType = "aog.relation.DefaultRelation";
		/**
		 * The type of relation used in the biclusters
		 */
		@Option(name = "-bcRelationType", usage = "The full class name of the relation used in the biclusters")
		public String bcRelationType = "aog.relation.DefaultRelation";

		/**
		 * The type of ICombiner in the grammar
		 */
		@Option(name = "-combinerType", usage = "The full class name of the combiner in the grammar")
		public String combinerType = "aog.relation.DefaultCombiner";

		/**
		 * The type of neighborhood judge
		 */
		@Option(name = "-neighborhoodJudgeType", usage = "The full class name of the neighborhood judge")
		public String neighborhoodJudgeType = "aog.learn.bc.DefaultNeighborhoodJudge";

		/*
		 * Parameters for distance-based neighborhood judgment
		 */
		/**
		 * Whether we iteratively grow the maximal distance in learning
		 */
		@Option(name = "-growDistNeighborhood", usage = "Whether we iteratively grow the maximal distance in learning")
		public boolean growDistNeighborhood = false;
		/**
		 * The initial maximal distance allowed between two patterns for
		 * distance-based neighborhood judgment
		 */
		@Option(name = "-initMaxPatternDist", usage = "The initial maximal distance for distance-based neighborhood judgment")
		public double initMaxPatternDist = 0;
		/**
		 * The step-size of growing the maximal distance
		 */
		@Option(name = "-stepSizeOfMaxPatternDist", usage = "The step-size of growing the maximal distance")
		public double stepSizeOfMaxPatternDist = 1;
		/**
		 * The maximal possible distance between two patterns in a sample
		 */
		@Option(name = "-finalMaxPatternDist", usage = "The maximal distance for distance-based neighborhood judgment")
		public double finalMaxPatternDist = 13;

		/*
		 * Biclustering parameters
		 */
		/**
		 * The percentage of seeds to be used for biclustering in
		 * bestBiClustering(). In principle, the larger this number is, the
		 * better quality the learned grammar might have, and the slower the
		 * algorithm is.
		 */
		@Option(name = "-ratioOfSeedsForBiClustering", usage = "The percentage of seeds to be used for biclustering")
		public double ratioOfSeedsForBiClustering = 1.0;

		/**
		 * The maximally allowed ratio of the real value to the predicted value
		 * of a cell in a multiplicative coherent bi-cluster. If the ratio is
		 * higher than this, we think that cell may involve overlapping
		 * bi-clusters and a warning is issued.
		 */
		@Option(name = "-mcbcMaxRatioToPrediction", usage = "The maximally expected ratio of the real vs. the predicted value of a cell in biclustering")
		public double mcbcMaxRatioToPrediction = 4;

		/**
		 * A bicluster with posterior gain less than this will be discarded.
		 */
		@Option(name = "-minPosteriorGain", usage = "A bicluster with posterior gain less than this will be discarded")
		public double minPosteriorGain = 0;

		/**
		 * Whether instance conflicts are allowed in the bicluster. A conflict
		 * exists if there are two bigrams (say, A, B) in the bicluster such
		 * that an appearance of A and an appearance of B involve the same
		 * pattern instance (i.e., either <a,b>-<a,c> or <b,a>-<c,a>).
		 */
		@Option(name = "-allowConflict", usage = "Whether instance conflicts are allowed in the bicluster")
		public boolean allowConflict = true;

		/*
		 * log prior gain by learning from a new bicluster is: alpha * (dl2 *
		 * sum + dl3 * A * B - dl1 * (A + B) - const), where sum is the element
		 * sum of the bicluster, A and B are the number of rows and columns.
		 */
		/**
		 * A parameter of the grammar prior probability. Used to adjust the
		 * importance of the prior in learning. The larger it is, the more
		 * important the prior is (and usually the more generalized the learning
		 * is).
		 */
		@Option(name = "-alpha", usage = "A parameter of the grammar prior probability")
		public double alpha = 0.5;
		/**
		 * A parameter of the grammar prior probability. It should be the number
		 * of bits needed for each child of an OR node.
		 */
		@Option(name = "-dl1", usage = "A parameter of the grammar prior probability")
		public double dl1 = 2;
		/**
		 * A parameter of the grammar prior probability. It should be the number
		 * of bits needed for each child of an AND node.
		 */
		@Option(name = "-dl2", usage = "A parameter of the grammar prior probability")
		public double dl2 = 3;
		/**
		 * A parameter of the grammar prior probability. Ideally it is the sum
		 * of the sizes of all the contexts in the expression-context (EC)
		 * matrix of the bicluster, with a discount caused by the sparsity of
		 * EC. In reality, it is hard to estimate and hence we use a constant.
		 * XXX this part is to be refined...
		 */
		@Option(name = "-dl3", usage = "A parameter of the grammar prior probability")
		public double dl3 = 0;

		/*
		 * Parameters of beam search
		 */
		/**
		 * The max size of the beam
		 */
		@Option(name = "-beamSize", usage = "The max size of the beam")
		public int beamSize = 1;
		/**
		 * A bicluster with a posterior gain less than the highest posterior
		 * gain minus this threshold is discarded from the beam.
		 */
		@Option(name = "-beamRelativeThreshold", usage = "The threshold of pruning beam elements")
		public double beamRelativeThreshold = Double.POSITIVE_INFINITY;
		/**
		 * Only put conflicting biclusters into the beam. Two biclusters
		 * conflict iff. they have overlapping or they share at least one
		 * instance (= they lead to different reductions of one or more
		 * instances).
		 */
		@Option(name = "-beamConflictingBcOnly", usage = "Only put conflicting biclusters into the beam")
		public boolean beamConflictingBcOnly = true;
	}

	public static Parameters param = new Parameters();
}
