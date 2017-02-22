package aog.sample;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import aog.TerminalPattern;

public class Corpus {
	public Sample[] samples;
	public TerminalPattern terms[];

	public Corpus() {
	}

	public Corpus(String filename) throws FileNotFoundException {
		readFromFile(filename);
	}

	public Corpus(Sample[] sentences, TerminalPattern[] terms) {
		this.samples = sentences;
		this.terms = terms;
	}

	/**
	 * Collect the terminal patterns used in the samples and assign to
	 * {@link #terms}.
	 */
	public void collectTermsFromSamples() {
		HashSet<TerminalPattern> termSet = new HashSet<>();
		for (Sample s : samples) {
			for (PatternInstance pi : s.elements) {
				termSet.add((TerminalPattern) pi.type);
			}
		}
		terms = termSet.toArray(new TerminalPattern[termSet.size()]);
	}

	/**
	 * Import a pre-processed corpus. The file format is as follows:
	 * <p>
	 * <li>Terminal pattern list: the first line is the number of terminal
	 * patterns; then the list of all the individual terminal patterns in the
	 * corpus, one per line.</li><br>
	 * <li>A blank line.</li><br>
	 * <li>Sample list: the first line is the number of samples; then the list
	 * of all the samples, one per line. Each sample is a sequence of terminal
	 * patterns (represented by their 0-based line numbers in the term list) and
	 * their parameters (in a pair of brackets, separated by space), e.g.,
	 * "1 [0.1 1.2] 5 [1.5 3.3]".</li>
	 * 
	 * @param filename
	 *            the name of the file containing the pre-processed corpus
	 * @throws FileNotFoundException
	 */
	public void readFromFile(String filename) throws FileNotFoundException {
		// read in terms
		Scanner s = new Scanner(new BufferedReader(new FileReader(filename)));
		int len = 0;
		if (s.hasNext())
			len = s.nextInt();
		terms = new TerminalPattern[len];
		s.nextLine(); // go to the 2nd line
		for (int i = 0; i < len; i++) {
			terms[i] = new TerminalPattern(s.nextLine());
		}

		s.nextLine();

		// read in samples
		len = 0;
		if (s.hasNext())
			len = s.nextInt();
		samples = new Sample[len];
		s.nextLine(); // go to the 2nd line
		for (int i = 0; i < len; i++) {
			samples[i] = new Sample();
			Scanner ss = new Scanner(s.nextLine());
			while (ss.hasNext()) {
				TerminalPattern t = terms[ss.nextInt()];
				Scanner sss = new Scanner(ss.findInLine("\\[[^\\]]*\\]"));
				sss.useDelimiter("[\\s\\[\\]]+");
				ArrayList<Double> parameters = new ArrayList<>();
				while (sss.hasNext()) {
					parameters.add(sss.nextDouble());
				}
				sss.close();
				PatternInstance pi = new PatternInstance(t, parameters,
						samples[i]);
				samples[i].elements.add(pi);
			}
			ss.close();
		}
		s.close();
	}

	/**
	 * Save the corpus to a file. The file format is the same as described in
	 * {@link #readFromFile(String)}.
	 * 
	 * @param filename
	 * @throws IOException
	 */
	public void saveToFile(String filename) throws IOException {
		// store terms
		FileWriter fw = new FileWriter(filename);
		fw.write(terms.length + "\n");
		for (int i = 0; i < terms.length; i++) {
			fw.write(terms[i] + "\n");
		}

		fw.write("\n");

		HashMap<TerminalPattern, Integer> termMap = new HashMap<>();
		for (int i = 0; i < terms.length; i++) {
			termMap.put(terms[i], i);
		}

		// store samples
		fw.write(samples.length + "\n");
		for (Sample s : samples) {
			for (PatternInstance pi : s.elements) {
				fw.write(termMap.get(pi.type) + " [");
				for (Double param : pi.parameters) {
					fw.write(param + " ");
				}
				fw.write("] ");
			}
			fw.write("\n");
		}

		fw.close();
	}

	public String toString() {

		StringBuilder sb = new StringBuilder();
		for (Sample s : samples) {
			sb.append(s.toString() + "\n");
		}

		return sb.toString();

	}

	// public void saveToReadableFile(String filename) {
	// try {
	// FileWriter fw = new FileWriter(filename);
	//
	// for (int i = 0; i < sentences.length; i++) {
	// for (Sample.Element j = sentences[i].firstElem; j != null; j = j.next) {
	// fw.write(j.sym + " ");
	// }
	// fw.write("\n");
	// }
	//
	// fw.close();
	// } catch (IOException e) {
	// e.printStackTrace();
	// }
	// }

	/*
	 * The following methods are used for cross-validation
	 */
	/**
	 * Split the corpus to n corpora of (roughly) equal size. Each corpus
	 * contains consecutive samples in the original corpus. Notice that the new
	 * Corpus objects share the terms and sentences with the original Corpus
	 * object.
	 * 
	 * @param n
	 * @return
	 */
	public Corpus[] split(int n) {
		Corpus[] re = new Corpus[n];
		int size = samples.length / n;
		for (int i = 0; i < n - 1; i++) {
			re[i] = new Corpus(Arrays.copyOfRange(samples, i * size, (i + 1)
					* size), terms);
		}
		re[n - 1] = new Corpus(Arrays.copyOfRange(samples, (n - 1) * size,
				samples.length), terms);
		return re;
	}

	/**
	 * Split the corpus to n corpora of (roughly) equal size. Neighboring
	 * samples in the original corpus are put into different corpora. Notice
	 * that the new Corpus objects share the terms and sentences with the
	 * original Corpus object.
	 * 
	 * @param n
	 * @return
	 */
	public Corpus[] split2(int n) {
		ArrayList<ArrayList<Sample>> split = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			split.add(new ArrayList<Sample>());
		}
		for (int i = 0; i < samples.length; i++) {
			split.get(i % n).add(samples[i]);
		}

		Corpus[] re = new Corpus[n];
		for (int i = 0; i < n; i++) {
			re[i] = new Corpus(split.get(i).toArray(new Sample[0]), terms);
		}
		return re;
	}

	/**
	 * Merge this Corpus and the input Corpus into a new corpus. The two Corpus
	 * to be merged must have the same value for the {@code terms} field. Notice
	 * that the new Corpus object shares the terms and sentences with the
	 * original two Corpus objects.
	 * 
	 * @param c2
	 * @return
	 */
	public Corpus merge(Corpus c2) {
		if (terms != c2.terms)
			return null;
		Sample[] sent = Arrays.copyOf(samples, samples.length
				+ c2.samples.length);
		for (int i = samples.length; i < sent.length; i++) {
			sent[i] = c2.samples[i - samples.length];
		}
		return new Corpus(sent, terms);
	}

	// /**
	// * Make a copy of the Corpus, where all the sentences and terms are
	// * duplicated.
	// *
	// * @return
	// */
	// protected Corpus copy() {
	// return copyExceptPart(0, 0);
	// }
	//
	// /**
	// * Make a copy of a part of the Corpus, where the sentences outside the
	// * specified index range are duplicated and all the terms are duplicated.
	// *
	// * @param from
	// * inclusive start index.
	// * @param to
	// * exclusive end index.
	// * @return
	// */
	// protected Corpus copyExceptPart(int from, int to) {
	// if (from < 0)
	// from = 0;
	// if (to > sentences.length)
	// to = sentences.length;
	//
	// TerminalPattern[] t = new TerminalPattern[terms.length];
	// HashMap<TerminalPattern, TerminalPattern> termMap = new
	// HashMap<TerminalPattern, TerminalPattern>(
	// (int) (terms.length / 0.75) + 1);
	// for (int i = 0; i < t.length; i++) {
	// t[i] = new TerminalPattern(terms[i].obj);
	// termMap.put(terms[i], t[i]);
	// }
	// Sample[] sent = new Sample[sentences.length - (to - from)];
	// for (int i = 0; i < from; i++) {
	// if (sentences[i].firstElem == null)
	// sent[i] = new Sample(null);
	// else {
	// Pattern firstSym = termMap.get(sentences[i].firstElem.sym);
	// firstSym.appearanceNum++;
	// Sample.Element first = new Sample.Element(firstSym);
	// Sample.Element p2 = first;
	// for (Sample.Element p = sentences[i].firstElem.next; p != null; p =
	// p.next) {
	// Pattern sym = termMap.get(p.sym);
	// sym.appearanceNum++;
	// p2.next = new Sample.Element(sym);
	// p2.next.prev = p2;
	// p2 = p2.next;
	// }
	// sent[i] = new Sample(first);
	// }
	// }
	// for (int i = to; i < sentences.length; i++) {
	// if (sentences[i].firstElem == null)
	// sent[i - to + from] = new Sample(null);
	// else {
	// Pattern firstSym = termMap.get(sentences[i].firstElem.sym);
	// firstSym.appearanceNum++;
	// Sample.Element first = new Sample.Element(firstSym);
	// Sample.Element p2 = first;
	// for (Sample.Element p = sentences[i].firstElem.next; p != null; p =
	// p.next) {
	// Pattern sym = termMap.get(p.sym);
	// sym.appearanceNum++;
	// p2.next = new Sample.Element(sym);
	// p2.next.prev = p2;
	// p2 = p2.next;
	// }
	// sent[i - to + from] = new Sample(first);
	// }
	// }
	// return new Corpus(sent, t);
	// }

}
