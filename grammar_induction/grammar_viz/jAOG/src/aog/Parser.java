package aog;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import aog.relation.ICombiner;
import aog.relation.INoisyRelation;
import aog.relation.IRelation;
import aog.sample.Corpus;
import aog.sample.PatternInstance;
import aog.sample.Sample;
import aog.util.Triple;
import aog.util.Util;

/**
 * A parser using dynamic programming. Note that we assume the grammar is a DAG,
 * i.e., it contains no recursive rule. TODO handle recursive rules
 * 
 * @author Kewei Tu
 * 
 */
public class Parser {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {
		Grammar g = new Grammar("tmp/output.g");
		Parser p = new Parser(g);
		Corpus c = new Corpus("tmp/test.cp");

		for (Sample s : c.samples) {
			System.out.println(p.parse(s, false));
		}
	}

	protected Grammar grammar;

	/**
	 * Whether to get the probability of the best parse (Viterbi parsing) or the
	 * sample (marginalizing the parse)
	 */
	protected boolean viterbi = true;

	/**
	 * The ordering of the nodes in the grammar in accordance with the partial
	 * order child-parent relation
	 */
	protected ArrayList<Pattern> patternQueue;

	/**
	 * A map from a node in the grammar to a set of candidate partial parses
	 * rooted at the node. The ordering of the nodes are the same as
	 * {@link #patternQueue}. Each candidate partial parse consists of 1) the
	 * set of pattern instances in the sample that are covered by this node in
	 * the parse, 2) the parameters of this node in the parse, and 3) the score
	 * of this node in the parse.
	 */
	protected Map<Pattern, Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>>> patternParses;

	public Parser(Grammar g) {
		setGrammar(g);
	}

	public void setGrammar(Grammar g) {
		grammar = g;

		HashMap<Pattern, int[]> childCount = new HashMap<>();
		for (AndNode and : g.andNodes) {
			childCount.put(and, new int[] { and.children.size() });
		}
		for (OrNode or : g.orNodes) {
			childCount.put(or, new int[] { or.children.size() });
		}

		patternQueue = new ArrayList<>();
		for (TerminalPattern t : g.terms) {
			patternQueue.add(t);
		}

		int i = 0;
		while (i < patternQueue.size()) {
			for (Pattern p : patternQueue.get(i).parents) {
				int[] count = childCount.get(p);
				count[0]--;
				if (count[0] == 0)
					patternQueue.add(p);
			}
			i++;
		}
	}

	/**
	 * @param sample
	 * @param viterbi
	 *            Whether to get the probability of the best parse (Viterbi
	 *            parsing) or the sample (marginalizing the parse)
	 * @param coverAll
	 *            Whether the parse is required to cover all the pattern
	 *            instances in the sample
	 * @return
	 */
	public double parse(Sample sample, boolean viterbi, boolean coverAll) {
		this.viterbi = viterbi;

		patternParses = new HashMap<>();
		for (int i = 0; i < patternQueue.size(); i++) {
			Pattern p = patternQueue.get(i);
			Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>> parses = new ArrayList<>();

			if (p instanceof TerminalPattern) { // TerminalPattern
				TerminalPattern t = (TerminalPattern) p;
				for (PatternInstance e : sample.elements) {
					if (((TerminalPattern) e.type).hasEqualObject(t)) {
						Set<PatternInstance> set = new HashSet<>();
						set.add(e);
						parses.add(new Triple<>(set, e.parameters, 1d));
					}
				}
			} else if (p instanceof AndNode) { // AndNode
				AndNode a = (AndNode) p;
				if (INoisyRelation.class.isAssignableFrom(grammar.relationType)) {
					// TODO
					System.err.println("[Error] not implemented");
					System.exit(1);
				}
				Collection<Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double>> combinations = traverseParseCombinationsOfAndChildren(
						a, 0, null);
				ICombiner combiner = a.pcRelation;
				for (Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double> comb : combinations) {
					parses.add(new Triple<>(comb.a, combiner
							.getParentFromChildren(comb.b), comb.c));
				}
			} else { // OrNode
				OrNode o = (OrNode) p;

				double Z = 0;
				for (double w : o.weights) {
					Z += w;
				}

				for (int j = 0; j < o.children.size(); j++) {
					Pattern child = o.children.get(j);
					double weight = o.weights.get(j);
					Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>> childParses = patternParses
							.get(child);
					for (Triple<Set<PatternInstance>, ArrayList<Double>, Double> parse : childParses) {
						double newScore = parse.c * weight / Z;
						addParse(parses, parse.a, parse.b, newScore);
					}
				}
			}

			patternParses.put(p, parses);
		}

		Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>> rootParses = patternParses
				.get(grammar.startSymbol);
		double rootScore = 0;
		for (Triple<Set<PatternInstance>, ArrayList<Double>, Double> parse : rootParses) {
			if (coverAll && parse.a.size() != sample.elements.size())
				continue;

			// TODO if coverAll=false and the parse doesn't cover the whole
			// sample, apply a penalty to the score

			if (viterbi) {
				if (rootScore < parse.c)
					rootScore = parse.c;
			} else {
				rootScore += parse.c;
			}
		}
		return rootScore;
	}

	protected void addParse(
			Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>> parses,
			Set<PatternInstance> newSet, ArrayList<Double> newParam,
			double newScore) {
		// check duplication
		for (Triple<Set<PatternInstance>, ArrayList<Double>, Double> p2 : parses) {
			if (Util.approxEquals(p2.b, newParam) && p2.a.equals(newSet)) {
				if (viterbi) {
					if (p2.c < newScore)
						p2.c = newScore;
				} else {
					p2.c += newScore;
				}

				/*
				 * There should be no more duplication, so we are done
				 */
				return;
			}
		}

		// add
		parses.add(new Triple<>(newSet, newParam, newScore));
	}

	/**
	 * Try all combinations of the parses of the children of the AndNode. The
	 * children before {@code childIndex} have been traversed and the combined
	 * parses are stored in {@code partialResults}. Return the valid
	 * combinations.
	 * 
	 * @param and
	 * @param childIndex
	 * @param partialResults
	 */
	protected Collection<Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double>> traverseParseCombinationsOfAndChildren(
			AndNode and,
			int childIndex,
			Collection<Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double>> partialResults) {
		// TODO process INoisyRelation

		Pattern child = and.children.get(childIndex);
		Collection<Triple<Set<PatternInstance>, ArrayList<Double>, Double>> curParses = patternParses
				.get(child);
		if (curParses.isEmpty())
			return new ArrayList<>();

		if (childIndex == 0) {
			Collection<Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double>> initPartialResults = new ArrayList<>();
			for (Triple<Set<PatternInstance>, ArrayList<Double>, Double> parse : curParses) {
				ArrayList<ArrayList<Double>> childParams = new ArrayList<>();
				childParams.add(parse.b);
				initPartialResults.add(new Triple<>(parse.a, childParams,
						parse.c));
			}

			if (and.children.size() == 1)
				return initPartialResults;
			else
				return traverseParseCombinationsOfAndChildren(and, 1,
						initPartialResults);
		} else {
			Collection<Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double>> newPartialResults = new ArrayList<>();
			for (Triple<Set<PatternInstance>, ArrayList<Double>, Double> curParse : curParses) {
				for (Triple<Set<PatternInstance>, ArrayList<ArrayList<Double>>, Double> preParse : partialResults) {
					// if the relation holds and no overlap, then combine them
					boolean relHold = true;
					for (int i = 0; i < childIndex; i++) {
						IRelation rel = and.ccRelations[childIndex][i];
						if (rel != null
								&& !rel.hasRelation(child, curParse.b,
										and.children.get(i), preParse.b.get(i))) {
							relHold = false;
							break;
						}
						rel = and.ccRelations[i][childIndex];
						if (rel != null
								&& !rel.hasRelation(and.children.get(i),
										preParse.b.get(i), child, curParse.b)) {
							relHold = false;
							break;
						}
					}

					if (relHold && Collections.disjoint(curParse.a, preParse.a)) {
						Set<PatternInstance> newSet = new HashSet<>(curParse.a);
						newSet.addAll(preParse.a);
						ArrayList<ArrayList<Double>> newParams = new ArrayList<>(
								preParse.b);
						newParams.add(curParse.b);
						double newScore = curParse.c * preParse.c;
						newPartialResults.add(new Triple<>(newSet, newParams,
								newScore));
					}
				}
			}

			if (childIndex + 1 == and.children.size()
					|| newPartialResults.isEmpty()) {
				return newPartialResults;
			} else {
				return traverseParseCombinationsOfAndChildren(and,
						childIndex + 1, newPartialResults);
			}
		}
	}

	/**
	 * Viterbi parsing
	 * 
	 * @param sample
	 * @return
	 */
	public double parse(Sample sample) {
		return parse(sample, true, true);

	}

	/**
	 * @param sample
	 * @param viterbi
	 *            Whether to get the probability of the best parse (Viterbi
	 *            parsing) or the sample (marginalizing the parse)
	 * @return
	 */
	public double parse(Sample sample, boolean viterbi) {
		return parse(sample, viterbi, true);

	}
}
