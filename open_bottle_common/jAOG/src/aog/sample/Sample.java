package aog.sample;

import java.util.ArrayList;

public class Sample {
	public ArrayList<PatternInstance> elements = new ArrayList<>();

	// summation at or nodes for the probability

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for (PatternInstance e : elements) {
			sb.append(e + " ");
		}
		sb.deleteCharAt(sb.length() - 1);
		sb.append("]");
		return sb.toString();
	}

}
