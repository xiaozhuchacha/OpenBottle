package aog.sample;

import java.util.ArrayList;

import aog.Pattern;

public class PatternInstance {

	public Pattern type;
	public ArrayList<Double> parameters;

	/**
	 * temporary variable used in Sampler
	 */
	ArrayList<Double> offset;

	public Sample containingSample;

	public PatternInstance() {
	}

	public PatternInstance(Pattern type, ArrayList<Double> parameters,
			Sample containingSample) {
		this.type = type;
		this.parameters = parameters;
		this.containingSample = containingSample;
	}

	public PatternInstance parent = null;
	public ArrayList<PatternInstance> children = new ArrayList<>();

	public PatternInstance(Pattern type, ArrayList<Double> parameters,
			Sample containingSample, PatternInstance[] children) {
		this.type = type;
		this.parameters = parameters;
		this.containingSample = containingSample;

		for (PatternInstance child : children) {
			this.children.add(child);
			child.parent = this;
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(type + "[");
		for (Double param : parameters) {
			sb.append(param + " ");
		}
		if (!parameters.isEmpty())
			sb.deleteCharAt(sb.length() - 1);
		sb.append("]");
		return sb.toString();
	}
}
