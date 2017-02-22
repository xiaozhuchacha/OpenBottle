package aog.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Util {

	/**
	 * Sample from a discrete distribution
	 * 
	 * @param distribution
	 *            the discrete distribution, which can be unnormalized.
	 * @return
	 */
	public static int discreteSampling(double[] distribution) {
		// compute the cumulative distribution function
		double[] cd = new double[distribution.length + 1];
		cd[0] = 0;
		int i;
		for (i = 0; i < distribution.length; i++) {
			cd[i + 1] = cd[i] + distribution[i];
		}

		// sample
		double rand = random() * cd[i];
		int ind = Arrays.binarySearch(cd, 0, i, rand);
		if (ind < 0)
			ind = -ind - 2;
		return ind;
	}

	/**
	 * A random number generator. We can fix the seed for easier debugging.
	 */
	public static Random random = new Random();

	/**
	 * Same as {@link Math#random()} but with our own random number generator.
	 * 
	 * @return
	 */
	public static double random() {
		return random.nextDouble();
	}

	/**
	 * The threshold for {@link #approxEquals(double, double)} and
	 * {@link #approxEquals(ArrayList, ArrayList)}.
	 */
	public static double precision = 1e-5;

	/**
	 * Check if the two numbers are approximately equal.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static boolean approxEquals(double a, double b) {
		return Math.abs(a - b) <= precision;
	}

	/**
	 * Check if the two vectors are approximately equal.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static boolean approxEquals(ArrayList<Double> a, ArrayList<Double> b) {
		if (a.size() != b.size())
			return false;

		for (int i = 0; i < a.size(); i++) {
			if (Math.abs(a.get(i) - b.get(i)) > precision)
				return false;
		}
		return true;
	}

	/**
	 * @param x
	 * @return x * log(x)
	 */
	public final static double xlogx(double x) {
		assert x >= 0;
		return x == 0 ? 0 : x * Math.log(x);
	}

}
