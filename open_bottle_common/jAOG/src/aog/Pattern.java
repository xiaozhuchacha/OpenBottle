package aog;

import java.util.ArrayList;

/**
 * Either a basic or a composite pattern
 * 
 * @author Kewei Tu
 * 
 */
public abstract class Pattern {
	public int id;

	private static int idCounter = 0;

	public Pattern() {
		id = idCounter++;
	}

	public ArrayList<Pattern> parents = new ArrayList<Pattern>();

}
