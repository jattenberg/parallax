/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Base class for setting and using an annealing schedule on the learning rate
 * 
 * @author jattanberg
 * 
 */
public abstract class AnnealingSchedule {

	final double initialLearningRate;

	public AnnealingSchedule(double initialLearningRate) {
		checkArgument(
				initialLearningRate > 0 && !Double.isNaN(initialLearningRate)
						&& !Double.isInfinite(initialLearningRate),
				"initial learning rate must be finite and positive, given: %s",
				initialLearningRate);

		this.initialLearningRate = initialLearningRate;
	}

	/**
	 * Returns the current value of the learning rate based on the current
	 * "epoch" and "dimension"
	 * 
	 * @param epoch
	 *            the current epoch
	 * @param dimension
	 *            the dimension along which the learning rate is to be annealed
	 * @return
	 */
	public abstract double learningRate(int epoch, int dimension);

	/**
	 * Receive a report from an optimizer about the effect of the specified
	 * learning rate in the specified epoch and return <code>true</code> if the
	 * update producing the error should be accepted or rejected.
	 * 
	 * may be used to ignore extreme loss cases
	 * 
	 * @param epoch
	 *            Training epoch.
	 * @param error
	 *            Training error.
	 */
	public boolean considerLoss(int epoch, double error) {
		return error > 10.e-10;
	}

	public double getInitialLearningRate() {
		return initialLearningRate;
	}
}
