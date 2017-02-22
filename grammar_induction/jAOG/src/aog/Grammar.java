package aog;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;
import java.util.regex.MatchResult;

import aog.relation.ICombiner;
import aog.relation.IRelation;

public class Grammar {
	public TerminalPattern[] terms;
	// protected Map<Object, TerminalPattern> termMap;

	// public TerminalPattern getTerminal(Object o) {
	// return termMap.get(o);
	// }

	public ArrayList<AndNode> andNodes;

	/**
	 * including the start symbol
	 */
	public ArrayList<OrNode> orNodes;

	/**
	 * For an ideal grammar, from the start symbol s we should be able to reach
	 * all the AND nodes. This is not always true, although those unreachable
	 * AND nodes won't be used either in parsing or in generation.
	 */
	public Pattern startSymbol;

	public Class<?> relationType;

	public Class<?> combinerType;

	/**
	 * Create a grammar with only terminals.
	 * 
	 * @param terms
	 */
	public Grammar(TerminalPattern[] terms) {
		this.terms = terms;
		// termMap = new HashMap<Object, TerminalPattern>();
		// for (int i = 0; i < terms.length; i++) {
		// TerminalPattern term = terms[i];
		// termMap.put(term.obj, term);
		// }
		andNodes = new ArrayList<AndNode>();
		orNodes = new ArrayList<OrNode>();
	}

	/**
	 * Create a grammar by reading from a file.
	 * 
	 * @param filename
	 * @throws FileNotFoundException
	 * @throws ClassNotFoundException
	 */
	public Grammar(String filename) throws Exception {
		readFromFile(filename);
	}

	public Grammar() {
		andNodes = new ArrayList<AndNode>();
		orNodes = new ArrayList<OrNode>();
		// termMap = new HashMap<Object, TerminalPattern>();
	}

	/**
	 * The file format:
	 * <p>
	 * "Terminal#" followed by the number of terminals<br>
	 * "AndNode#" followed by the number of And nodes<br>
	 * "OrNode#" followed by the number of Or nodes<br>
	 * "StartSymbol" followed by the ID of the Or node representing the start
	 * symbol<br>
	 * Relation type<br>
	 * "Terminals"<br>
	 * A list of terminal patterns, one per line, in the format of "ID String"<br>
	 * "AndNodes"<br>
	 * A list of And nodes, one per line, in the format of
	 * "ID [list of child_IDs][list of relations between children][combiner]"
	 * where the list elements are separated by space<br>
	 * "OrNodes"<br>
	 * A list of Or nodes, one per line, in the format of
	 * "ID [list of child_IDs] [list of probabilities]" where the list elements
	 * are separated by space<br>
	 * 
	 * @param filename
	 * @throws IOException
	 */
	public void saveToFile(String filename) throws IOException {
		int ind = 0;
		HashMap<Pattern, Integer> symMap = new HashMap<>();
		for (int i = 0; i < terms.length; i++) {
			symMap.put(terms[i], ind);
			ind++;
		}

		for (AndNode and : andNodes) {
			symMap.put(and, ind);
			ind++;
		}

		for (OrNode or : orNodes) {
			symMap.put(or, ind);
			ind++;
		}

		FileWriter fw = new FileWriter(filename);
		fw.write("Terminal# " + terms.length + "\n");
		fw.write("AndNode# " + andNodes.size() + "\n");
		fw.write("OrNode# " + orNodes.size() + "\n");
		fw.write("StartSymbol " + symMap.get(startSymbol) + "\n");
		fw.write(relationType.getName() + "\n");
		fw.write(combinerType.getName() + "\n");

		fw.write("Terminals\n");
		for (TerminalPattern t : terms) {
			fw.write(symMap.get(t) + "\t" + t.obj + "\n");
		}

		fw.write("AndNodes\n");
		for (AndNode and : andNodes) {
			fw.write(symMap.get(and) + "\t[");

			for (Pattern p : and.children) {
				fw.write(symMap.get(p) + " ");
			}
			fw.write("]");

			fw.write("[");
			for (int i = 0; i < and.children.size(); i++) {
				for (int j = 0; j < and.children.size(); j++) {
					IRelation rel = and.ccRelations[i][j];
					if (rel != null)
						fw.write("<" + i + "," + j + ">:("
								+ rel.getStringFromRelation() + ") ");
				}
			}
			fw.write("]");

			fw.write("[");
			fw.write(and.pcRelation.getStringFromCombiner());
			fw.write("]");

			fw.write("\n");
		}

		fw.write("OrNodes\n");
		for (OrNode or : orNodes) {
			fw.write(symMap.get(or) + "\t[");
			for (Pattern p : or.children) {
				fw.write(symMap.get(p) + " ");
			}
			fw.write("] [");
			for (double w : or.weights) {
				fw.write(w + " ");
			}
			fw.write("]\n");
		}

		fw.close();
	}

	/**
	 * File format is the same as described in {@link #saveToFile(String)}.
	 * 
	 * @param filename
	 * @throws Exception
	 */
	public void readFromFile(String filename) throws Exception {
		Scanner s = new Scanner(new BufferedReader(new FileReader(filename)));

		s.next();
		int nTerms = s.nextInt();
		terms = new TerminalPattern[nTerms];

		s.next();
		int nAnd = s.nextInt();
		andNodes = new ArrayList<AndNode>(nAnd);

		s.next();
		int nOr = s.nextInt();
		orNodes = new ArrayList<OrNode>(nOr);

		HashMap<Integer, AndNode> andMap = new HashMap<>();
		HashMap<Integer, TerminalPattern> terminalMap = new HashMap<>();
		HashMap<Integer, OrNode> orMap = new HashMap<>();

		s.next();
		int startSymbolId = s.nextInt();

		relationType = Class.forName(s.next());
		combinerType = Class.forName(s.next());

		// read in all of the terminal nodes
		s.next();
		for (int i = 0; i < terms.length; i++) {
			int id = s.nextInt();
			// System.out.println("Adding Terminal " + id +"\n");
			terms[i] = new TerminalPattern(s.nextLine().trim());
			terminalMap.put(id, terms[i]);
		}

		// initialize all of the And nodes
		s.next();
		for (int i = 0; i < nAnd; i++) {
			int id = s.nextInt();
			// System.out.println("Adding And " + id +"\n");
			s.nextLine();
			AndNode and = new AndNode();
			andNodes.add(and);
			andMap.put(id, and);
		}

		// initialize all of the Or nodes
		s.next();
		for (int i = 0; i < nOr; i++) {
			int id = s.nextInt();
			// System.out.println("Adding Or " + id +"\n");
			s.nextLine();
			OrNode or = new OrNode();
			orNodes.add(or);
			orMap.put(id, or);
		}

		s.close();

		s = new Scanner(new BufferedReader(new FileReader(filename)));

		for (int i = 0; i < nTerms + 8; ++i) {
			s.nextLine();
		}

		for (int i = 0; i < nAnd; i++) {

			int id = s.nextInt();
			AndNode and = andMap.get(id);

			Scanner ss = new Scanner(s.findInLine("\\[[^\\]]*\\]"));
			ss.useDelimiter("[\\s\\[\\]]+");

			// System.out.println("And node is " + id + "\n");

			while (ss.hasNext()) {

				int childId = ss.nextInt();

				// System.out.println("child is " + childId + "\n");
				if (andMap.get(childId) != null) {
					and.addChild(andMap.get(childId));
				} else if (orMap.get(childId) != null) {
					and.addChild(orMap.get(childId));
				} else if (terminalMap.get(childId) != null) {
					and.addChild(terminalMap.get(childId));
				} else {
					System.out.println("Error finding child node!");
				}
			}

			ss.close();

			// ccRelations
			IRelation[][] ccRelations = new IRelation[and.children.size()][and.children
					.size()];
			String str = s.findInLine("\\[[^\\]]*\\]");
			str = str.substring(1, str.length() - 1);
			ss = new Scanner(str);
			while (ss.hasNext()) {
				ss.findInLine("<(\\d+),(\\d+)>:\\(([^\\)]*)\\)");
				MatchResult mr = ss.match();
				int ind1 = Integer.parseInt(mr.group(1));
				int ind2 = Integer.parseInt(mr.group(2));
				String relStr = mr.group(3);
				IRelation r = (IRelation) relationType.newInstance();
				r.getRelationFromString(relStr);
				ccRelations[ind1][ind2] = r;
			}
			ss.close();
			and.ccRelations = ccRelations;

			// pcRelation
			str = s.findInLine("\\[[^\\]]*\\]");
			str = str.substring(1, str.length() - 1);
			and.pcRelation = (ICombiner) combinerType.newInstance();
			and.pcRelation.getCombinerFromString(str);
		}

		s.next();
		while (s.hasNext()) {
			int id = s.nextInt();
			// System.out.println("Or node is " + id + "\n");
			OrNode or = orMap.get(id);
			Scanner ss = new Scanner(s.findInLine("\\[[^\\]]*\\]"));
			ss.useDelimiter("[\\s\\[\\]]+");

			ArrayList<Pattern> children = new ArrayList<>();
			while (ss.hasNext()) {

				int childId = ss.nextInt();
				// System.out.println("child is " + childId + "\n");
				if (andMap.get(childId) != null) {
					children.add(andMap.get(childId));
				} else if (orMap.get(childId) != null) {
					children.add(orMap.get(childId));
				} else if (terminalMap.get(childId) != null) {
					children.add(terminalMap.get(childId));
				} else {
					System.out.println("Error finding child node!");
				}
			}
			ss.close();
			ss = new Scanner(s.findInLine("\\[[^\\]]*\\]"));
			ss.useDelimiter("[\\s\\[\\]]+");
			ArrayList<Double> weights = new ArrayList<>();
			while (ss.hasNext()) {
				weights.add(ss.nextDouble());
			}
			ss.close();
			or.addChildren(children, weights);
		}

		s.close();

		// termMap = new HashMap<Object, TerminalPattern>();
		// for (TerminalPattern t : terms) {
		// termMap.put(t.obj, t);
		// }

		if (andMap.get(startSymbolId) != null) {
			startSymbol = andMap.get(startSymbolId);
		} else if (orMap.get(startSymbolId) != null) {
			startSymbol = orMap.get(startSymbolId);
		} else if (terminalMap.get(startSymbolId) != null) {
			startSymbol = terminalMap.get(startSymbolId);
		} else {
			System.out.println("Error: Start symbol not found");
		}

	}

	/**
	 * Remove nodes that are not reachable from the start symbol
	 */
	public void removeRedundancy() {
		// TODO
	}

	/**
	 * Simplify the grammar by merging parent-child And-nodes or Or-nodes and
	 * removing single-child nodes. TODO finish
	 */
	public void simplify() {
		for (Iterator<OrNode> iter = orNodes.iterator(); iter.hasNext();) {
			OrNode n = iter.next();
			// remove single-child nodes
			if (n.children.size() == 1) {
				Pattern c = n.children.get(0);
				iter.remove();
				for (Pattern p : n.parents) {
					if (p instanceof AndNode) {
						AndNode pa = (AndNode) p;
						int ind = pa.children.indexOf(n);
						pa.children.set(ind, c);
					} else if (p instanceof OrNode) {
						OrNode po = (OrNode) p;
						int ind = po.children.indexOf(n);
						po.children.set(ind, c);
					}
				}
			}

			// TODO merge child Or-nodes
		}

		// TODO process And-nodes
	}
}
