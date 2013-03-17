/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

/**
 * A multivariate distribution composed of many bernoulli probability distributions.
 * 
 * @see {@link <a href="http://en.wikipedia.org/wiki/Bernoulli_distribution">Bernoulli distribution on wikipedia</a>}
 */
public class BernoulliMultivariateDistribution extends AbstractDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -2618940953916576803L;
	
	/** The component univariate distributions comprizing the multivariate distribution. */
	private BernoulliDistribution[] dists;
	
	/** The alpha value used for laplace smoothing.*/
	private double alpha = 0;

	/**
	 * Instantiates a new bernoulli multivariate distribution.
	 *
	 * @param size the dimension of the input space
	 */
	public BernoulliMultivariateDistribution(int size) {
		super(size);
		initialize();
	}

	/**
	 * Instantiates a new bernoulli multivariate distribution.
	 *
	 * @param size the dimension of the input space
	 * @param alpha The alpha value used for laplace smoothing.
	 */
	public BernoulliMultivariateDistribution(int size, double alpha) {
		super(size);
		this.alpha = alpha;
		initialize();
	}

	/**
	 * Initialize the internal data structures. 
	 */
	private void initialize() {
		dists = new BernoulliDistribution[size];
		for (int i = 0; i < size; i++)
			dists[i] = new BernoulliDistribution(alpha);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#observe(int, double)
	 */
	@Override
	public void observe(int dimension, double value) {
		dists[dimension].observe(value);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#probability(int)
	 */
	@Override
	public double probability(int dimension) {
		return dists[dimension].probability(dimension);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#logProbability(int, double)
	 */
	@Override
	public double logProbability(int dimension, double value) {
		return logProbability(dimension);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#probability(int, double)
	 */
	@Override
	public double probability(int dimension, double value) {
		return probability(dimension);
	}

}
