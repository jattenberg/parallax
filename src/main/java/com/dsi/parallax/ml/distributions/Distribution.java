/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;

import java.io.Serializable;

/**
 * Interface describing multivariate probability distributions
 * 
 * @author jattenberg
 */
public interface Distribution extends Serializable {

	/**
	 * Observe a particular value along the given dimension.
	 * 
	 * @param dimension
	 *            the dimension of the observation
	 * @param value
	 *            the value observed
	 */
	public abstract void observe(int dimension, double value);

	/**
	 * Probability of encountering a value along a dimension.
	 * 
	 * @param dimension
	 *            the dimension of the observation
	 * @param value
	 *            the value value observed
	 * @return the probability of this observation
	 */
	public abstract double probability(int dimension, double value);

	/**
	 * Probability of observing 0 along a certain dimension
	 * 
	 * @param dimension
	 *            the dimension of the observation
	 * @return the probability of observiting a 0 in this dimension
	 */
	public abstract double probability(int dimension);

	/**
	 * Log-Probability of encountering a value along a dimension.
	 * 
	 * @param dimension
	 *            the dimension of the observation
	 * @param value
	 *            the value value observed
	 * @return the log-probability of this observation
	 */
	public abstract double logProbability(int dimension, double value);

	/**
	 * Log Probability of observing 0 along a certain dimension
	 * 
	 * @param dimension
	 *            the dimension of the observation
	 * @return the log probability of observiting a 0 in this dimension
	 */
	public abstract double logProbability(int dimension);

	/**
	 * Probabilities of 0's across all dimensions
	 * 
	 * @return the probabilities of observing 0 across all dimensions
	 */
	public abstract double[] probabilities();

	/**
	 * Log Probabilities of 0's across all dimensions
	 * 
	 * @return the log probabilities of observing 0 across all dimensions
	 */
	public abstract double[] logProbabilities();

}
