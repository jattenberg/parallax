/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

/**
 * The base class for multi-dimensional probability distributions. maintains the
 * number of covariates to be considered in the distribution and implementations
 * of {@link #logProbabilities()}, {@link #probabilities()}, and
 * {@link #logProbability(int)}
 * 
 */
public abstract class AbstractDistribution implements Distribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 12345L;

	/**
	 * The number of covariates in the input space of the probability
	 * distribution
	 */
	protected final int size;

	/**
	 * Instantiates a new abstract distribution.
	 * 
	 * @param size
	 *            number of covariates in the input space of the probability
	 *            distribution
	 */
	protected AbstractDistribution(int size) {
		this.size = size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#logProbability(int)
	 */
	@Override
	public double logProbability(int dimension) {
		return Math.log(probability(dimension));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#probabilities()
	 */
	@Override
	public double[] probabilities() {
		double[] out = new double[size];
		for (int i = 0; i < size; i++)
			out[i] = probability(i);
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#logProbabilities()
	 */
	@Override
	public double[] logProbabilities() {
		double[] out = new double[size];
		for (int i = 0; i < size; i++)
			out[i] = logProbability(i);
		return out;
	}
}
