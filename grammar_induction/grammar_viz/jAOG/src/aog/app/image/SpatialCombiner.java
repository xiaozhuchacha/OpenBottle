package aog.app.image;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Scanner;

import aog.relation.AbstractCombiner;
import aog.relation.ICombiner;

public class SpatialCombiner extends AbstractCombiner implements ICombiner {

	/**
	 * The offset of the parent parameters from the center of the child
	 * parameters.
	 */
	public double dx = 0, dy = 0;

	public SpatialCombiner(double dx, double dy) {
		this.dx = dx;
		this.dy = dy;
	}

	public SpatialCombiner() {
	}

	@Override
	public ArrayList<Double> getParentFromChildren(
			Collection<ArrayList<Double>> paramOfChildren) {
		double x0 = 0, y0 = 0;
		for (ArrayList<Double> param : paramOfChildren) {
			x0 += param.get(0);
			y0 += param.get(1);
		}
		x0 /= paramOfChildren.size();
		y0 /= paramOfChildren.size();
		x0 += dx;
		y0 += dy;

		ArrayList<Double> re = new ArrayList<>(2);
		re.add(x0);
		re.add(y0);
		return re;
	}

	@Override
	public String getStringFromCombiner() {
		return dx + "," + dy;
	}

	@Override
	public void getCombinerFromString(String s) {
		Scanner sc = new Scanner(s);
		sc.useDelimiter("[,\\s]+");
		dx = sc.nextDouble();
		dy = sc.nextDouble();
		sc.close();
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof SpatialCombiner) {
			SpatialCombiner b = (SpatialCombiner) obj;
			return dx == b.dx && dy == b.dy;
		} else
			return false;
	}

	@Override
	public int hashCode() {
		// convert possible -0 to +0
		if (dx == 0)
			dx = 0.0d;
		if (dy == 0)
			dy = 0.0d;

		return Double.valueOf(dx).hashCode() * 31
				+ Double.valueOf(dy).hashCode();
	}

}
