/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

public class ExponentialAnnealingSchedule extends
		DimensionAgnosticAnnealingSchedule {

	private final double exponentialBase;

	public ExponentialAnnealingSchedule(double initialLearningRate,
			double exponentialBase) {
		super(initialLearningRate);
		checkArgument(exponentialBase > 0 && exponentialBase <= 1,
				"exponentialBase must be in (0, 1], given: %s", exponentialBase);
		this.exponentialBase = exponentialBase;
	}

	@Override
	public double learningRate(int epoch) {
		return initialLearningRate * Math.pow(exponentialBase, epoch);
	}

	public double getExponentialBase() {
		return exponentialBase;
	}

}
