/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.optimization.Optimizable;
//TODO: change interface to use some kind of updateable model and instances? 
/**
 * interface for optimization techniques that are based on the function 
 * gradient. the primary example is SGD. Note that all optimization is 
 * on minimization. for maximization, the negative of the objective 
 * function should be used. 
 * 
 * optimized function is typically "cleaned up" before it is used. this 
 * incoporated any unapplied regularization terms
 * 
 * @author jattenberg
 *
 */
public interface GradientStochasticOptimizer {
	/**
	 * update the set of parameters according to a gradient after observing some loss
	 * @param function function with parameters theta_{t-1}, parameters used to generate gradient and loss. to be optimized
	 *     and gradient \grad f(x_t, theta_{t-1}) gradient used to optimize theta and loss f(x_t, theta_{t-1}), loss observed. 
	 * @return updated model, with  theta_t
	 */
	public Optimizable update(Optimizable function);
	
	/**
	 * clean up any outstanding regularization. Vital since lazy regularization may not shirnk
	 * parameters until they are touched. 
	 * @param function- function wiht parameters to be updated via regularization
	 * @return updated model, with theta_t
	 */
	public Optimizable cleanup(Optimizable function);
}
