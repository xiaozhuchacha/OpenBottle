package aog;

public class TerminalPattern extends Pattern {
	public Object obj;

	public TerminalPattern(Object o) {
		super();
		obj = o;
	}

	@Override
	public String toString() {
		return obj.toString();
	}

	public boolean hasEqualObject(TerminalPattern t1) {
		return obj.equals(t1.obj);
	}
}
