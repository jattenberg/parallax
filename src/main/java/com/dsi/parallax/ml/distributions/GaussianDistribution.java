/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;

/**
 * A simple multivariate gaussian distribution with diagonal covariance. <br>
 * {@link <a href="http://en.wikipedia.org/wiki/Normal_distribution">Normal Distribution</a>}
 * for more info
 */
public class GaussianDistribution extends AbstractDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 2125108809249088118L;

	/** Constituent univariate gaussians. */
	private UnivariateGaussianDistribution[] dists;

	/**
	 * Instantiates a new gaussian distribution.
	 * 
	 * @param size
	 *            number of dimensions in the multivariate distribution
	 */
	public GaussianDistribution(int size) {
		super(size);
		initialize();
	}

	/**
	 * Initialize the internal data structures in order to start accumulating
	 * data
	 */
	private void initialize() {
		dists = new UnivariateGaussianDistribution[size];
		for (int i = 0; i < size; i++)
			dists[i] = new UnivariateGaussianDistribution();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#observe(int, double)
	 */
	@Override
	public void observe(int dimension, double value) {
		dists[dimension].observe(value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#probability(int)
	 */
	@Override
	public double probability(int dimension) {
		return probability(dimension, 1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#logProbability(int,
	 * double)
	 */
	@Override
	public double logProbability(int dimension, double value) {
		return Math.log(dists[dimension].probability(value));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#probability(int, double)
	 */
	@Override
	public double probability(int dimension, double value) {
		return dists[dimension].probability(value);
	}
}
