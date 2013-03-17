package com.parallax.ml.util.bounds;

public class Bounds {

	private final LowerBound lowerBound;
	private final UpperBound upperBound;

	/**
	 * @param lowerBound
	 * @param upperBound
	 */
	public Bounds(LowerBound lowerBound, UpperBound upperBound) {
		this.lowerBound = lowerBound;
		this.upperBound = upperBound;
	}

	public LowerBound getLowerBound() {
		return lowerBound;
	}

	public UpperBound getUpperBound() {
		return upperBound;
	}

	/**
	 * return an array of bounds for numeric values for floating point options
	 * 
	 * @return an array of [lower bound, upper bound]
	 */
	public double[] numericValues() {
		double[] out = new double[2];
		if (lowerBound instanceof GreaterThanOrEqualsValueBound)
			out[0] = lowerBound.getBoundValue();
		else
			out[0] = lowerBound.getBoundValue() + (1. / 1000000.);
		if (upperBound instanceof LessThanOrEqualsValueBound)
			out[1] = upperBound.getBoundValue();
		else
			out[1] = upperBound.getBoundValue() - (1. / 1000000.);
		return out;
	}

	/**
	 * return an array of bounds for numeric values for integer valued options
	 * 
	 * @return an array of [lower bound, upper bound]
	 */
	public int[] integerValues() {
		int[] out = new int[2];
		if (lowerBound instanceof GreaterThanOrEqualsValueBound)
			out[0] = (int) lowerBound.getBoundValue();
		else
			out[0] = (int) lowerBound.getBoundValue() + 1;
		if (upperBound instanceof LessThanOrEqualsValueBound)
			out[1] = (int) upperBound.getBoundValue();
		else
			out[1] = (int) upperBound.getBoundValue() - 1;

		return out;
	}

}
