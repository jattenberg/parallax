/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.model;

import com.dsi.parallax.ml.target.RealValuedTarget;

/**
 * The interface for regression models- models that predict an 
 * unconstrained real value.
 *
 * @param <M> The concrete type of model- used for method chaining.
 */
public interface RegressionModel<M extends RegressionModel<M>> extends Model<RealValuedTarget, M> {

}
