/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic.anneal;

/**
 * Class for using a constant learning rate
 * 
 * @author jattenberg
 * 
 */
public class ConstantAnnealingSchedule extends
		DimensionAgnosticAnnealingSchedule {

	public ConstantAnnealingSchedule(double learningRate) {
		super(learningRate);
	}

	/**
	 * Returns a constant learning rate independent of the epoch
	 */
	@Override
	public double learningRate(int epoch) {
		return initialLearningRate;
	}
}
