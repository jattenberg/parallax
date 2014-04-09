/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;

/**
 * A bernoulli probability distribution.
 * 
 * @see {@link <a href="http://en.wikipedia.org/wiki/Bernoulli_distribution">Bernoulli distribution on wikipedia</a>}
 */
public class BernoulliDistribution extends AbstractUnivariateDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 7632140063893632124L;

	/** The number of successes observed */
	private double successes = 0;
	/** The number of trials performed */
	private double trials = 0;

	/** The alpha value used for laplace smoothing. */
	private double alpha = 0;

	/**
	 * Instantiates a new bernoulli distribution.
	 */
	public BernoulliDistribution() {

	}

	/**
	 * Instantiates a new bernoulli distribution.
	 * 
	 * @param alpha
	 *            the alpha value used for laplace smoothing
	 */
	public BernoulliDistribution(double alpha) {
		this.alpha = alpha;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.UnivariateDistribution#observe(double)
	 */
	@Override
	public void observe(double value) {
		successes += value;
		trials++;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.distributions.UnivariateDistribution#probability(double)
	 */
	@Override
	public double probability(double datum) {
		return (alpha + successes) / (2 * alpha + trials);
	}

	/**
	 * Sets the alpha value used for laplace smoothing.
	 * 
	 * @param alpha
	 *            the new alpha value used for laplace smoothing
	 */
	public void setAlpha(int alpha) {
		this.alpha = alpha;
	}
}
