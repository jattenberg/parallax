/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.util.MLUtils;

/**
 * Multivariate histogram distribution constructed from many univariate
 * histogram distributions
 */
public class HistogramDistribution extends AbstractDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5380285982530067735L;

	/** The number of bins in each histogram. */
	private int bins;

	/** The univariate histograms */
	private HistogramUnivariateDistribution[] hists;

	/**
	 * Instantiates a new histogram distribution.
	 * 
	 * @param dimension
	 *            the number of dimensions being modeled
	 * @param bins
	 *            the number of bins in the histograms
	 */
	public HistogramDistribution(int dimension, int bins) {
		super(dimension);
		checkArgument(bins > 1, "bins must be > 1, given: %s", bins);
		this.bins = bins;
		hists = new HistogramUnivariateDistribution[dimension];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#observe(int, double)
	 */
	@Override
	public void observe(int dimension, double value) {
		if (null == hists[dimension])
			hists[dimension] = new HistogramUnivariateDistribution(bins);
		hists[dimension].observe(value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.Distribution#probability(int, double)
	 */
	@Override
	public double probability(int dimension, double value) {
		if (null == hists[dimension])
			return 0;
		else
			return hists[dimension].probability(value);
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
		double prob = probability(dimension, value);
		return MLUtils.floatingPointEquals(0, prob) ? 0 : Math.log(prob);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.distributions.AbstractDistribution#logProbability(int)
	 */
	@Override
	public double logProbability(int dimension) {
		return logProbability(dimension, 1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.distributions.AbstractDistribution#probabilities()
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
	 * @see
	 * com.parallax.ml.distributions.AbstractDistribution#logProbabilities()
	 */
	@Override
	public double[] logProbabilities() {
		double[] out = new double[size];
		for (int i = 0; i < size; i++)
			out[i] = logProbability(i);
		return out;
	}
}
