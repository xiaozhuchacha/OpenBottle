package aog.util;

public class Triple<A, B, C> {
	public A a;
	public B b;
	public C c;

	public Triple(A a, B b, C c) {
		this.a = a;
		this.b = b;
		this.c = c;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj.getClass() == getClass()) {
			@SuppressWarnings("unchecked")
			Triple<A, B, C> t2 = (Triple<A, B, C>) obj;
			return (a.equals(t2.a)) && (b.equals(t2.b)) && (c.equals(t2.c));
		}
		return false;
	}

	@Override
	public int hashCode() {
		return (a.hashCode() * 31 + b.hashCode()) * 31 + c.hashCode();
	}

	@Override
	public String toString() {
		return "(" + a + "," + b + "," + c + ")";
	}

}
