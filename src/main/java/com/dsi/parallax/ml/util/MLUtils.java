/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import com.dsi.parallax.ml.vector.LinearVector;
import org.apache.commons.lang.RandomStringUtils;

import java.io.*;
import java.lang.reflect.Array;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static com.google.common.base.Preconditions.checkArgument;
import static org.apache.commons.math.special.Gamma.logGamma;

public class MLUtils {
	public static final double LOG2 = Math.log(2);
	public static final double ROOT2 = Math.sqrt(2);
	public static final double SMALL = 1e-6;
	private static final int LONGEST_WORD = 15;
	private final static int MIN_QSORT_SIZE = 7;
	public static final Random GENERATOR = new Random(
			System.currentTimeMillis());

	private MLUtils() {
	}

	private final static String cleanText(String tmp, int maxlen,
			boolean ignoreStopWords) {

		StringTokenizer tok = new StringTokenizer(tmp,
				" +.,~\\<>\\$?!:;(){}|-0123456789\b\t\n\f\r\"\'\\\\/\\=\\&\\%\\_");
		StringBuilder buff = new StringBuilder();
		while (tok.hasMoreTokens()) {
			String out = tok.nextToken();
			if (out.length() < 2 || out.length() > maxlen
					|| (ignoreStopWords && StopWordSet.stopwords.contains(out)))
				continue;
			buff.append(out + " ");
		}
		return buff.toString();
	}

	public final static String cleanText(String tmp, boolean ignoreStopWords) {
		return cleanText(tmp, LONGEST_WORD, ignoreStopWords);
	}

	public final static String grammify(String tmp, int size) {
		if (tmp == null || size < 1)
			throw new IllegalArgumentException(
					"input must not be null and size must be >= 1");
		StringBuffer buff = new StringBuffer();
		int start = 0;

		while (start + size <= tmp.length()) {
			buff.append(start == 0 ? "" : " "
					+ tmp.substring(start, start + size));
			start++;
		}
		return buff.toString();
	}

	public final static String stringSplitter(String tmp, int size) {
		if (tmp == null || size < 1)
			throw new IllegalArgumentException(
					"input must not be null and size must be >= 1");
		StringBuffer buff = new StringBuffer();
		int start = 0;
		String delim = "";
		while (start + size <= tmp.length()) {
			buff.append(delim + tmp.substring(start, start + size));
			delim = " ";
			start += size;
		}
		if (start != tmp.length())
			buff.append(delim + tmp.substring(start));
		return buff.toString();
	}

	public static String reverseStringSplitter(String tmp, int size) {
		if (tmp == null || size < 1)
			throw new IllegalArgumentException(
					"input must not be null and size must be >= 1");
		StringBuffer buff = new StringBuffer();

		int start = tmp.length() - 1;
		String delim = "";
		while (start - size >= 0) {
			buff.append(delim + tmp.substring(start - size, start));
			delim = " ";
			start -= size;
		}
		if (start != 0)
			buff.append(delim + tmp.substring(0, start));
		return buff.toString();
	}

	/**
	 * finds the appropriate bin for a value amongst the values associated with
	 * quantiles of a value range.
	 * 
	 * @param value
	 * @param quantiles
	 * @return
	 */
	public static final int quantileBin(double value, double[] quantiles) {

		for (int i = 0; i < quantiles.length; i++)
			if (value <= quantiles[i])
				return i;
		return quantiles.length;
	}

	public static double max(double... input) {
		double out = input[0];
		for (int i = 1; i < input.length; i++) {
			if (input[i] > out)
				out = input[i];
		}
		return out;
	}

	public static double min(double... input) {
		double out = input[0];
		for (int i = 1; i < input.length; i++) {
			if (input[i] < out)
				out = input[i];
		}
		return out;
	}

	/**
	 * Returns the correlation coefficient of two double vectors.
	 * 
	 * @param y1
	 *            double vector 1
	 * @param y2
	 *            double vector 2
	 * @param n
	 *            the length of two double vectors
	 * @return the correlation coefficient
	 */
	public static final double correlation(double y1[], double y2[], int n) {
		

		int i;
		double av1 = 0.0, av2 = 0.0, y11 = 0.0, y22 = 0.0, y12 = 0.0, c;

		if (n <= 1) {
			return 1.0;
		}
		for (i = 0; i < n; i++) {
			av1 += y1[i];
			av2 += y2[i];
		}
		av1 /= n;
		av2 /= n;
		for (i = 0; i < n; i++) {
			y11 += (y1[i] - av1) * (y1[i] - av1);
			y22 += (y2[i] - av2) * (y2[i] - av2);
			y12 += (y1[i] - av1) * (y2[i] - av2);
		}
		if (y11 * y22 == 0.0) {
			c = 1.0;
		} else {
			c = y12 / Math.sqrt(Math.abs(y11 * y22));
		}

		return c;
	}

	/**
	 * returns the value such that q% are above or below, depending on the value
	 * highest
	 * 
	 * @param input
	 * @param q
	 * @param highest
	 * @return
	 * @throws Exception
	 * 
	 *             TODO: make linear time
	 */
	public static final double findQuantile(Collection<Double> input, double q,
			boolean highest) throws Exception {
		if (q <= 0 || q >= 1)
			throw new IllegalArgumentException(
					"quantile must be between 0 and 1. given: " + q);
		List<Double> tmp = new ArrayList<Double>(input.size());
		for (double d : input)
			tmp.add(d);

		Collections.sort(tmp);
		int index = highest ? (int) Math.floor(tmp.size() * (1. - q))
				: (int) Math.ceil(tmp.size() * q);
		return tmp.get(index);

	}

	public static final double lnGamma(double input) {
		return logGamma(input);
	}

	public static final int hashLong(Long x) {
		return (int) (x ^ (x >>> 32));
	}

	public static final int hashDouble(double x) {
		return hashLong(Double.doubleToLongBits(x));
	}

	public final static <T> Collection<T> sample(Iterable<T> source, int size) {
		List<T> reservoir = new ArrayList<T>(size);

		Iterator<T> iterator = source.iterator();
		for (int i = 0; i < size; i++)
			reservoir.add(iterator.next());

		int index = size + 1;
		while (iterator.hasNext()) {
			int j = getRandomNumberInRange(0, index, true);
			index++;
			if (j < size)
				reservoir.set(j, iterator.next());
		}

		return reservoir;
	}

	public final static int getRandomNumberInRange(int min, int max,
			boolean inclusive) {

		int diff = max - min;
		if (inclusive)
			diff += 1;
		return GENERATOR.nextInt(diff) + min;
	}

	public static int nextDiscrete(List<Double> a, double sum) {
		double b = 0;
		double r = GENERATOR.nextDouble() * sum;
		for (int i = 0; i < a.size(); i++) {
			b += a.get(i);
			if (b > r) {
				return i;
			}
		}
		return a.size() - 1;
	}

	public static int nextDiscrete(List<Double> a) {
		return nextDiscrete(a, 1);
	}

	/**
	 * 
	 * @param prob
	 *            //prob for next example
	 * @param N
	 *            //number of steps
	 * @return //prob of occurring at least once
	 */
	public static double pOneX(double prob, double N) {
		return 1. - Math.pow(1. - prob, N);
	}

	public final static int maxIndex(double[] probs) {
		int index = 0;
		double max = probs[0];
		for (int i = 1; i < probs.length; i++) {
			if (probs[i] > max) {
				max = probs[i];
				index = i;
			}
		}
		return index;
	}

	public static void writeObject(Object o, String filename) throws Exception {
		ObjectOutput oo;
		oo = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(
				filename)));
		oo.writeObject(o);
		oo.close();

	}

	public static Object readObject(String filename) throws Exception {
		Object o = null;

		ObjectInput oo = new ObjectInputStream(new GZIPInputStream(
				new FileInputStream(filename)));

		o = oo.readObject();
		oo.close();

		return o;
	}

	public static double[] normalize(double[] input) {
		double[] out = new double[input.length];
		double tot = 0;
		for (int i = 0; i < input.length; i++) {
			tot += input[i];
			out[i] = input[i];
		}
		if (tot != 0)
			for (int i = 0; i < input.length; i++)
				out[i] /= tot;
		return out;
	}

	/**
	 * puts a prob onto the intervat [-1,1]
	 * 
	 * @param prob
	 */
	public static double probToSVMInterval(double prob) {
		return 2. * prob - 1.;
	}

	/**
	 * puts interval [-1,1] to a prob
	 * 
	 * @param svmScore
	 * @return
	 */
	public static double svmIntervalToProb(double svmScore) {
		return (svmScore + 1.) / 2.;
	}

	public static final double inverseSquareRoot(float x) {
		float xhalf = 0.5f * x;
		int i = Float.floatToIntBits(x); // get bits for floating value
		i = 0x5f375a86 - (i >> 1); // gives initial guess y0
		x = Float.intBitsToFloat(i); // convert bits back to float
		x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases
		// accuracy
		return x;
	}

	public static <T extends Comparable<T>> T slowselect(T[] arr, int k) {
		@SuppressWarnings("unchecked")
		T[] tmp = (T[]) Array.newInstance(arr[0].getClass(), arr.length);
		System.arraycopy(arr, 0, tmp, 0, arr.length);
		quicksort(tmp);
		return tmp[k];
	}

	/**
	 * selects the k^th smallest from an array of Ts.
	 * 
	 * @param <T>
	 * @param arr
	 * @param k
	 * @return
	 */
	public static <T extends Comparable<T>> T quickselect(T[] arr, int k) {
		checkArgument(k >= 0 && k < arr.length,
				"index must be within size of array, [%s, %s). given: %s", 0,
				arr.length, k);
		return quickselect(arr, 0, arr.length - 1, k);
	}

	private static <T extends Comparable<T>> T quickselect(T[] arr, int left,
			int right, int k) {
		while (left != right) {
			int rand = GENERATOR.nextInt(right - left) + left;
			int newPivotIndex = partition(arr, left, right, rand);
			if (newPivotIndex == k)
				return arr[newPivotIndex];
			else if (k < newPivotIndex)
				right = newPivotIndex;
			else {
				// k-= pivotDist;
				left = newPivotIndex + 1;
			}
		}
		return arr[left];
	}

	/**
	 * sorts the elements in an array via quicksort
	 * 
	 * @param <T>
	 * @param arr
	 */
	public static <T extends Comparable<T>> void quicksort(T[] arr) {
		quicksort(arr, 0, arr.length - 1);
	}

	private static <T extends Comparable<T>> void quicksort(T[] arr, int left,
			int right) {
		if (right > left) {
			int rand = GENERATOR.nextInt(right - left) + left;
			int newPivotIndex = partition(arr, left, right, rand);
			quicksort(arr, left, newPivotIndex - 1);
			quicksort(arr, newPivotIndex + 1, right);
		}
	}

	private static <T extends Comparable<T>> int partition(T[] arr, int left,
			int right, int pivotIndex) {
		T pivotValue = arr[pivotIndex];
		swapElements(arr, right, pivotIndex);
		int storeIndex = left;

		for (int i = left; i < right; i++) {
			if (arr[i].compareTo(pivotValue) < 0) {
				swapElements(arr, i, storeIndex);
				storeIndex++;
			}
		}
		swapElements(arr, right, storeIndex);
		return storeIndex;
	}

	private static <T> void swapElements(T[] arr, int first, int second) {
		T tmp = arr[first];
		arr[first] = arr[second];
		arr[second] = tmp;
	}

	/**
	 * Computes the entropy of the given array.
	 * 
	 * @param array
	 *            the array
	 * @return the entropy
	 */
	public static double entropy(double[] array) {

		double returnValue = 0, sum = 0;

		for (int i = 0; i < array.length; i++) {
			returnValue -= floatingPointEquals(array[i], 0) ? 0 : array[i]
					* Math.log(array[i]);
			sum += array[i];
		}
		if (MLUtils.floatingPointEquals(sum, 0)) {
			return 0;
		} else {
			return (returnValue + sum * Math.log(sum)) / (sum * MLUtils.LOG2);
		}
	}

	public static final double sum(double[] array) {
		double out = 0;
		for (double d : array)
			out += d;
		return out;
	}

	public static final double sum(Iterable<Double> values) {
		double out = 0;
		for (double d : values)
			out += d;
		return out;
	}

	public static final double fastLog2(double x) {
		int i = Float.floatToIntBits((float) x);
		double y = i;
		double f = (i & 0x007FFFFF) | (0x7e << 23);
		y *= 1.0 / (1 << 23);
		return y - 124.22544637f - 1.498030302f * f - 1.72587999f
				/ (0.3520887068f + f);
	}

	public static final double fastLog(double x) {
		return 0.69314718 * fastLog2(x);
	}

	public static final double fasterLog2(double x) {
		double y = Float.floatToIntBits((float) x);
		y *= 1.0 / (1 << 23);
		return y - 126.94269504f;
	}

	public static final double fasterLog(double x) {
		return 0.69314718 * fasterLog2(x);
	}

	public final static double infinityNorm(Collection<Double> w) {
		double max = Double.MIN_VALUE;
		for (double w_i : w)
			if (w_i > max)
				max = w_i;
		return max;
	}

	public final static double infinityNorm(double[] w) {
		double max = Double.MIN_VALUE;
		for (double w_i : w)
			if (w_i > max)
				max = w_i;
		return max;
	}

	public final static double oneNorm(Collection<Double> w) {
		double tot = 0.;
		for (double w_i : w)
			tot += Math.abs(w_i);
		return tot;
	}

	public final static double oneNorm(double[] w) {
		double tot = 0.;
		for (double w_i : w)
			tot += Math.abs(w_i);
		return tot;
	}

	public static final double twoNorm(Collection<Double> w) {
		double tot = 0.;
		for (double w_i : w)
			tot += Math.pow(w_i, 2.);
		return Math.sqrt(tot);
	}

	public static final double twoNorm(double[] w) {
		double tot = 0.;
		for (double w_i : w)
			tot += Math.pow(w_i, 2.);
		return Math.sqrt(tot);
	}

	public static final String randomString(int words) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < words; i++)
			sb.append(RandomStringUtils.randomAscii(6)).append(" ");
		return sb.toString();
	}

	public final static void qsortIncreasing(int[] x) {
		qsort(x, new IncreasingIntComp());
	}

	public final static void qsortDecreasing(int[] x) {
		qsort(x, new DecreasingIntComp());
	}

	public final static void isort(int[] x, CompareInt compare) {
		for (int i = 0; i < x.length; ++i) {
			int t = x[i];
			int j = i;
			for (; j > 0 && compare.lessThan(t, x[j - 1]); --j)
				x[j] = x[j - 1];
			x[j] = t;
		}
	}

	public final static void qsort(int[] x, CompareInt compare) {

		qsortPartial(x, 0, x.length - 1, compare, GENERATOR);
		isort(x, compare);
	}

	private static final void qsortPartial(int[] x, int lower, int upper,
			CompareInt compare, Random random) {
		if (upper - lower < MIN_QSORT_SIZE)
			return;
		swap(x, lower, lower + random.nextInt(upper - lower + 1));
		int t = x[lower];
		int i = lower;
		int j = upper + 1;
		while (true) {
			do {
				++i;
			} while (i <= upper && compare.lessThan(x[i], t));
			do {
				--j;
			} while (compare.lessThan(t, x[j]));
			if (i > j)
				break;
			swap(x, i, j);
		}
	}

	public final static void swap(int[] xs, int i, int j) {
		int temp = xs[i];
		xs[i] = xs[j];
		xs[j] = temp;
	}

	public interface CompareInt {
		public boolean lessThan(int a, int b);
	}

	public static class IncreasingIntComp implements CompareInt {

		@Override
		public boolean lessThan(int a, int b) {
			return a < b;
		}

	}

	public static class DecreasingIntComp implements CompareInt {

		@Override
		public boolean lessThan(int a, int b) {
			return b < a;
		}

	}

	public static class IncreasingIntDoubleArrComp implements CompareInt {
		private double[] arr;

		public IncreasingIntDoubleArrComp(double[] arr) {
			this.arr = arr;
		}

		@Override
		public boolean lessThan(int a, int b) {
			return arr[a] < arr[b];
		}
	}

	public static class DecreasingIntDoubleArrComp implements CompareInt {
		private double[] arr;

		public DecreasingIntDoubleArrComp(double[] arr) {
			this.arr = arr;
		}

		@Override
		public boolean lessThan(int a, int b) {
			return arr[a] > arr[b];
		}
	}

	public static final boolean floatingPointEquals(double a, double b) {
		return (a - b < SMALL) && (b - a < SMALL);
	}

	public static boolean floatingPointGreaterThan(double a, double b) {

		return (a - b > SMALL);
	}

	public static boolean floatingPointLessThanOrEquals(double a, double b) {

		return (a - b < SMALL);
	}

	public static boolean floatingPointLessThan(double a, double b) {

		return (b - a > SMALL);
	}

	public static final int[] sortGetIndexes(double[] values) {
		if (values == null || values.length == 0)
			throw new IllegalArgumentException(
					"values must be non-null, length > 0");
		int size = values.length;
		int[] indexes = new int[size];
		for (int i = 0; i < size; i++)
			indexes[i] = i;
		CompareInt cmp = new IncreasingIntDoubleArrComp(values);
		qsort(indexes, cmp);
		return indexes;
	}

	public static final double sech(double x) {
		return 2. * Math.exp(x) / (Math.exp(2. * x) + 1.);
	}

	public static final int longHash(long l) {
		return (int) (l ^ (l >>> 32));
	}

	public static final int doubleHash(double d) {
		return longHash(Double.doubleToLongBits(d));
	}

	public static final double boxMullerHash(String input) {
		int hash_one = Math.abs(input.hashCode());
		int hash_two = Math.abs((input + "_______two").hashCode());
		if (0 == hash_one || 0 == hash_two)
			return 0;
		double val_one = (double) hash_one / (double) Integer.MAX_VALUE;
		double val_two = (double) hash_two / (double) Integer.MAX_VALUE;
		return Math.sqrt(-2. * Math.log(val_one))
				* Math.cos(2. * Math.PI * val_two);
	}

	public static final double logAdd(double logX, double logY) {
		// 1. make X the max
		if (logY > logX) {
			double temp = logX;
			logX = logY;
			logY = temp;
		}
		// 2. now X is bigger
		if (logX == Double.NEGATIVE_INFINITY) {
			return logX;
		}
		// 3. how far "down" (think decibels) is logY from logX?
		// if it's really small (20 orders of magnitude smaller), then ignore
		double negDiff = logY - logX;
		if (negDiff < -20) {
			return logX;
		}
		// 4. otherwise use some nice algebra to stay in the log domain
		// (except for negDiff)
		return logX + java.lang.Math.log(1.0 + java.lang.Math.exp(negDiff));
	}

	public static boolean isNumeric(String str) {
		NumberFormat formatter = NumberFormat.getInstance();
		ParsePosition pos = new ParsePosition(0);
		formatter.parse(str, pos);
		return str.length() == pos.getIndex();
	}

	public static double euclidianDistance(LinearVector x, LinearVector y) {
		double out = 0;
		for (int x_i : x)
			out += Math.pow(x.getValue(x_i) * y.getValue(x_i), 2);
		return Math.sqrt(out);
	}

	/**
	 * Computes the Gini impurity of a histogram. For each item in the
	 * histogram, it is the probability that it is randomly assigned to the
	 * wrong category, given the frequency of the different categories. This is
	 * computed by looping over all the categories and multiplying the fraction
	 * of elements in that category (f_i) times the probability of choosing a
	 * different category (1 - f_i). That is:
	 * 
	 * sum_i f_i * (1 - f_i)
	 * 
	 * @return The Gini impurity of the given distribution.
	 */
	public static double giniImpurity(double[] counts) {
		double sum = sum(counts);
		if (sum == 0)
			return 0.;

		double totalImpurity = 0.;

		for (double count : counts) {
			double fraction = count / sum;
			totalImpurity += fraction * (1. - fraction);
		}

		return totalImpurity;
	}
}
