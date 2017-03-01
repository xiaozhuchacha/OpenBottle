package aog.util;

public class Pair<A, B> {
	public A a;
	public B b;

	public Pair(A a, B b) {
		this.a = a;
		this.b = b;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj.getClass() == getClass()) {
			@SuppressWarnings("unchecked")
			Pair<A, B> p2 = (Pair<A, B>) obj;
			return (a.equals(p2.a)) && (b.equals(p2.b));
		}
		return false;
	}

	@Override
	public int hashCode() {
		return a.hashCode() * 31 + b.hashCode();
	}

	@Override
	public String toString() {
		return "(" + a + "," + b + ")";
	}

}
