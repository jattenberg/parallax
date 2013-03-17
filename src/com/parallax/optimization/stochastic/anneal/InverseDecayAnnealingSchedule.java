/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

public class InverseDecayAnnealingSchedule extends
		DimensionAgnosticAnnealingSchedule {

	private final double annealingRate;

	public InverseDecayAnnealingSchedule(double initialLearningRate,
			double annealingRate) {
		super(initialLearningRate);
		checkArgument(annealingRate > 0 && !Double.isNaN(annealingRate)
				&& !Double.isInfinite(annealingRate),
				"annealing rate must be > 0 and finite, given %s",
				annealingRate);
		this.annealingRate = annealingRate;
	}

	@Override
	public double learningRate(int epoch) {
		return initialLearningRate / (1.0 + epoch / annealingRate);
	}

	public double getDecay() {
		return annealingRate;
	}
}
