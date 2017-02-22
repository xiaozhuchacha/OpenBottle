package aog.evaluate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import aog.Grammar;
import aog.Parser;
import aog.Pattern;
import aog.learn.bc.GrammarLearner;
import aog.sample.Corpus;
import aog.sample.PatternInstance;
import aog.sample.Sample;
import aog.sample.Sampler;

public class Evaluate {
	/*
	 * TODO Define a probability smoothing method that can be called by KL and
	 * perplexity estimation. It provides the zeroProb mode and background model
	 * mode (see the calculatePerplexity2/3 methods).
	 */

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {
		for (int i = 0; i < 10; i++) {
			Grammar g = new Grammar("tmp/ZZAnimal/MyAog/" + i + ".g");
			Corpus c = new Corpus("tmp/ZZAnimal/corpus_test" + i + ".cp");
			// System.out.println(calculatePerplexity(g, c, zeroProb));
			System.out.println(calculatePerplexity3(g, c, 1e-2, 1e-2 / 625));
			// double[] re = calculatePerplexity2(g, c);
			// System.out.println(re[0] + "\t" + re[1]);
		}
	}

	public static double[][] evaluateGrammar(Grammar gTrue, int nSamples,
			int nSamplesForEval, int nTrials, double detectionThreshold,
			int skipNum) {

		Grammar gramTest;
		double klDiv;
		double precision;
		double recall;

		int arrayLength = (int) Math.ceil((double) nSamples / skipNum) + 1;

		double[] numSamples = new double[arrayLength];
		double[] precisionArray = new double[arrayLength];
		double[] recallArray = new double[arrayLength];
		double[] klDivArray = new double[arrayLength];

		precisionArray[0] = 0;
		recallArray[0] = 0;
		klDivArray[0] = 0;
		numSamples[0] = 0;

		int ctr = 0;
		for (int i = 4000; i <= nSamples; i += skipNum) {

			ctr++;

			numSamples[ctr] = i;
			for (int j = 0; j < nTrials; ++j) {
				Corpus corpus = Sampler.sampleCorpusFromGrammar(gTrue, i);
				gramTest = GrammarLearner.learnGrammar(corpus);

				// Calculate the Precision
				precision = calculatePrecision(gTrue, gramTest,
						nSamplesForEval, detectionThreshold);
				precisionArray[ctr] += precision;
				System.out.println("Precision is " + precision);

				// Calculate the Recall
				recall = calculateRecall(gTrue, gramTest, nSamplesForEval,
						detectionThreshold);
				recallArray[ctr] += recall;
				System.out.println("Recall is " + recall);

				// Calculate the KL Divergence
				klDiv = calculateKLDivergence(gTrue, gramTest, nSamplesForEval,
						zeroProb);
				klDivArray[ctr] += klDiv;
				System.out.println("KL Divergence is " + klDiv);

				Corpus corpusTest = Sampler.sampleCorpusFromGrammar(gramTest,
						nSamplesForEval);
				Sample[] samplesTest = corpusTest.samples;
				Sample sample = samplesTest[0];
				System.out.println(sample);
			}
			precisionArray[ctr] /= nTrials;
			recallArray[ctr] /= nTrials;
			klDivArray[ctr] /= nTrials;

		}
		double[][] arrays = { precisionArray, recallArray, klDivArray,
				numSamples };
		return arrays;

	}

	/**
	 * Calculate the KL divergence between two grammars, KL Divergence = sum(pi
	 * log (pi / qi)), sampled from p (true grammar).
	 * 
	 * @param trueGrammar
	 * @param testGrammar
	 * @param nSamples
	 * @param zeroProb
	 *            if a probability is zero, and use this value to avoid infinity
	 * @return
	 */
	public static double calculateKLDivergence(Grammar trueGrammar,
			Grammar testGrammar, int nSamples, double zeroProb) {

		// Sample from the true grammar
		Corpus corpusTrue = Sampler.sampleCorpusFromGrammar(trueGrammar,
				nSamples);
		Sample[] samplesTrue = corpusTrue.samples;

		// Calculate the KL Divergence on the sample
		double KLDivergence = 0;

		Parser parser1 = new Parser(trueGrammar);
		Parser parser2 = new Parser(testGrammar);
		for (Sample sample : samplesTrue) {
			double p = parser1.parse(sample, false);
			double q = parser2.parse(sample, false);
			if (q == 0)
				q = zeroProb;
			KLDivergence += ((double) 1 / nSamples) * Math.log(p / q);
		}

		return KLDivergence;
	}

	/**
	 * Calculate Recall: sample from true grammar, detect with learned grammar.
	 * The detection counts if the probability is above the threshold value.
	 * 
	 * @param trueGrammar
	 * @param testGrammar
	 * @param nSamples
	 * @param threshold
	 * @return
	 */
	public static double calculateRecall(Grammar trueGrammar,
			Grammar testGrammar, int nSamples, double threshold) {

		// Sample from the true grammar
		Corpus corpusTrue = Sampler.sampleCorpusFromGrammar(trueGrammar,
				nSamples);
		Sample[] samplesTrue = corpusTrue.samples;
		int correctRecalls = 0;

		Parser parser = new Parser(testGrammar);
		for (Sample sample : samplesTrue) {
			double q = parser.parse(sample);
			// System.out.println(q);
			if (q > threshold) {
				correctRecalls++;
			}
		}

		return ((double) correctRecalls) / ((double) nSamples);
	}

	/**
	 * Calculate Precision: sample from learned grammar, detect with true
	 * grammar. The detection counts if the probability is above the threshold
	 * value.
	 * 
	 * @param trueGrammar
	 * @param testGrammar
	 * @param nSamples
	 * @param threshold
	 * @return
	 */
	public static double calculatePrecision(Grammar trueGrammar,
			Grammar testGrammar, int nSamples, double threshold) {

		// Sample from the true grammar
		Corpus corpusTest = Sampler.sampleCorpusFromGrammar(testGrammar,
				nSamples);
		Sample[] samplesTest = corpusTest.samples;
		int nDetected = 0;

		Parser parser = new Parser(trueGrammar);
		for (Sample sample : samplesTest) {
			double p = parser.parse(sample);

			// System.out.println(p);
			if (p > threshold) {
				nDetected++;
			}
		}

		return ((double) nDetected) / ((double) nSamples);

	}

	public static HashMap<Pattern, Integer> calculateHistogram(Sample[] samples) {

		HashMap<Pattern, Integer> histogram = new HashMap<>();

		// Corpus c = new Corpus();
		int nSamples = samples.length;

		for (int i = 0; i < nSamples; ++i) {

			Sample sample = samples[i];
			ArrayList<PatternInstance> elements = sample.elements;

			int nElements = elements.size();

			for (int j = 0; j < nElements; ++j) {
				PatternInstance node = elements.get(j);
				Pattern nodeType = node.type;
				Integer numInstances = histogram.get(nodeType);

				if (numInstances == null) {

					histogram.put(nodeType, 1);
				} else {

					histogram.put(nodeType, numInstances + 1);
				}

			}
		}

		return histogram;
	}

	/**
	 * Compute perplexity. Use zeroProb if the sample probability is less than
	 * zeroProb.
	 * 
	 * @param g
	 * @param testCorpus
	 * @param zeroProb
	 * @return
	 */
	public static double calculatePerplexity(Grammar g, Corpus testCorpus,
			double zeroProb) {
		Parser parser = new Parser(g);
		double logp = 0;
		for (Sample sample : testCorpus.samples) {
			double p = parser.parse(sample, false);
			if (p < zeroProb) {
				p = zeroProb;
			}
			logp += Math.log(p);
		}
		logp /= testCorpus.samples.length;
		return Math.exp(-logp);
	}

	/**
	 * Compute perplexity. Instead of using zeroProb, output the number of
	 * unparsable samples and the log likelihood of the parsable samples.
	 * 
	 * @param g
	 * @param testCorpus
	 * @return
	 */
	public static double[] calculatePerplexity2(Grammar g, Corpus testCorpus) {
		Parser parser = new Parser(g);
		double logp = 0;
		int zeroCount = 0;
		for (Sample sample : testCorpus.samples) {
			double p = parser.parse(sample, false);
			if (p == 0)
				zeroCount++;
			else
				logp += Math.log(p);
		}
		// logp /= testCorpus.samples.length;
		return new double[] { zeroCount, logp };
	}

	/**
	 * Compute perplexity. Assume a uniform background model that assigns
	 * bgModelProb to any sample. The total probability that the background
	 * model assigns to all possible samples is bgModelWeight.
	 * 
	 * @param g
	 * @param testCorpus
	 * @param bgModelWeight
	 * @param bgModelProb
	 * @return
	 */
	public static double calculatePerplexity3(Grammar g, Corpus testCorpus,
			double bgModelWeight, double bgModelProb) {
		Parser parser = new Parser(g);
		double logp = 0;
		for (Sample sample : testCorpus.samples) {
			double p = parser.parse(sample, false);
			double p2 = p * (1 - bgModelWeight) + bgModelProb;
			logp += Math.log(p2);
		}
		logp /= testCorpus.samples.length;
		return Math.exp(-logp);
	}

	/**
	 * In calculating KL or Perplexity, if a probability is zero, and use this
	 * value to avoid infinity
	 */
	protected static double zeroProb = 1e-5;
}