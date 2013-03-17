/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic;

import com.parallax.ml.vector.LinearVector;
import com.parallax.optimization.WeightedGradient;

/**
 * interface for optimization techniques that are based on the function 
 * gradient, including an explicit weight on each gradient value. 
 * @author jattenberg
 *
 */
public interface WeightSensitiveGradientStochasticOptimizer extends GradientStochasticOptimizer {
	public LinearVector update(LinearVector parameter, WeightedGradient gradient, double loss);
	public LinearVector updateModel(LinearVector parameter, WeightedGradient gradient, double loss);

}

