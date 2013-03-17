/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear.optimizable;

import com.parallax.ml.classifier.UpdateableClassifier;
import com.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

/**
 * The Interface describing behavior of {@link UpdateableClassifier}s that can
 * be optimized using general stochastic optimization procedures based on the
 * gradient of a loss function.
 * 
 * @param <C>
 *            the concrete type of classifier used for method chaining
 */
public interface GradientUpdateableClassifier<C extends GradientUpdateableClassifier<C>>
		extends UpdateableClassifier<C> {

	/**
	 * used for defining the optimization routine used for building models
	 * 
	 * @param optimizer
	 * @return
	 */
	public <O extends StochasticGradientOptimizationBuilder<O>> C setOptimizationBuilder(
			O optimizer);

	/**
	 * get the optimization routine used to train the model
	 * 
	 * @return
	 */
	public StochasticGradientOptimizationBuilder<?> getOptimizationBuilder();
}
