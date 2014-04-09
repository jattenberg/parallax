/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;

import com.dsi.parallax.ml.util.Histogram;

/**
 * A univariate distribution based on a histogram.
 */
public class HistogramUnivariateDistribution implements UnivariateDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -7461397573489731489L;

	/** The histogram used internally */
	private Histogram hist;

	/**
	 * Instantiates a new histogram univariate distribution.
	 * 
	 * @param bins
	 *            the number of bins in the histogram
	 */
	public HistogramUnivariateDistribution(int bins) {
		hist = new Histogram(bins);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.UnivariateDistribution#observe(double)
	 */
	@Override
	public void observe(double value) {
		hist.add(value);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.distributions.UnivariateDistribution#probability(double)
	 */
	@Override
	public double probability(double value) {
		return hist.closestBinProbability(value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.distributions.UnivariateDistribution#logProbability(double
	 * )
	 */
	@Override
	public double logProbability(double value) {
		return Math.log(hist.closestBinProbability(value));
	}

}
