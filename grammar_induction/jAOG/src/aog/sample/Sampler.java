package aog.sample;

import java.io.IOException;
import java.util.ArrayList;

import aog.AndNode;
import aog.Grammar;
import aog.OrNode;
import aog.Pattern;
import aog.TerminalPattern;
import aog.relation.IRelation;
import aog.util.Pair;
import aog.util.Util;

public class Sampler {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {

		assert (args.length == 3);

		String input_grammar_file = args[0];
		String output_file = args[1];

		int nSamples = Integer.parseInt(args[2]);

		Grammar g = new Grammar(input_grammar_file);

		System.out.println("Sampling...");
		Corpus c = sampleCorpusFromGrammar(g, nSamples);
		c.saveToFile(output_file);
		System.out.println("Done sampling");

		// Sample[] samples = c.samples;
		// for (int i = 0; i < 5; ++i) {
		// System.out.println(samples[i]);
		// }
		// System.out.println(c);

	}

	protected static IRelation nullRelation = null;

	protected static PatternInstance sampleTree(Pattern node) {

		if (node instanceof TerminalPattern) {
			TerminalPattern t = (TerminalPattern) node;
			return new PatternInstance(t, nullRelation.getDefaultParameter(t),
					null);

		} else if (node instanceof AndNode) {
			AndNode and = (AndNode) node;
			int n = and.children.size();
			PatternInstance[] children = new PatternInstance[n];
			ArrayList<ArrayList<Double>> paramOfChildren = new ArrayList<>();

			for (int i = 0; i < n; i++) {
				children[i] = sampleTree(and.children.get(i));

				ArrayList<Pair<IRelation, PatternInstance>> leftRels = new ArrayList<>();
				ArrayList<Pair<IRelation, PatternInstance>> rightRels = new ArrayList<>();
				for (int j = 0; j < i; j++) {
					IRelation rel = and.ccRelations[i][j];
					if (rel != null)
						leftRels.add(new Pair<>(rel, children[j]));

					rel = and.ccRelations[j][i];
					if (rel != null)
						rightRels.add(new Pair<>(rel, children[j]));
				}
				ArrayList<Double> newParam = nullRelation
						.updateParamFromRelations(children[i], leftRels,
								rightRels);

				ArrayList<Double> offset = new ArrayList<>();
				for (int j = 0; j < newParam.size(); j++) {
					double newp = newParam.get(j);
					double oldp = children[i].parameters.get(j);
					offset.add(newp - oldp);
				}
				children[i].parameters = newParam;
				children[i].offset = offset;
				paramOfChildren.add(newParam);
			}

			ArrayList<Double> param = and.pcRelation
					.getParentFromChildren(paramOfChildren);
			return new PatternInstance(and, param, null, children);

		} else if (node instanceof OrNode) {
			OrNode orNode = (OrNode) node;
			ArrayList<Pattern> children = orNode.children;
			ArrayList<Double> weights = orNode.weights;
			int nChildren = children.size();

			double[] weightsArray = new double[nChildren];
			for (int i = 0; i < nChildren; ++i) {
				weightsArray[i] = weights.get(i);
			}
			int index = Util.discreteSampling(weightsArray);

			Pattern child = children.get(index);
			return sampleTree(child);
		}

		return null;
	}

	protected static void collectLeaves(PatternInstance pi, Sample sample,
			ArrayList<Double> offset0) {
		if (pi.children.isEmpty()) {
			for (int i = 0; i < offset0.size(); i++) {
				double o = offset0.get(i);
				double p = pi.parameters.get(i);
				pi.parameters.set(i, p + o);
			}
			sample.elements.add(new PatternInstance(pi.type, pi.parameters,
					sample));
		} else {
			ArrayList<Double> offset = pi.offset;
			for (int i = 0; i < offset0.size(); i++) {
				double o0 = offset0.get(i);
				double o = offset.get(i);
				offset.set(i, o + o0);
			}

			for (PatternInstance c : pi.children) {
				collectLeaves(c, sample, offset);
			}
		}
	}

	public static Sample sampleFromGrammar(Grammar g) {
		PatternInstance root = sampleTree(g.startSymbol);

		int dim = root.parameters.size();
		ArrayList<Double> offset = new ArrayList<Double>(dim);
		ArrayList<Double> offset0 = new ArrayList<Double>(dim);
		for (int i = 0; i < dim; i++) {
			offset.add(0d);
			offset0.add(0d);
		}
		root.offset = offset;

		// Domain-specific adjustment of the root offset
		// offset.set(0, 5.5);
		// offset.set(1, 5.0);

		Sample sample = new Sample();
		collectLeaves(root, sample, offset0);

		return sample;
	}

	public static Corpus sampleCorpusFromGrammar(Grammar g, int nSamples) {
		try {
			nullRelation = (IRelation) g.relationType.newInstance();
		} catch (InstantiationException | IllegalAccessException e) {
			e.printStackTrace();
		}

		Corpus c = new Corpus();
		c.samples = new Sample[nSamples];

		for (int i = 0; i < nSamples; ++i) {
			c.samples[i] = sampleFromGrammar(g);
		}
		c.terms = g.terms;

		return c;
	}
}
