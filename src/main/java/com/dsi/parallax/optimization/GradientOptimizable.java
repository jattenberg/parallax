/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import com.dsi.parallax.ml.vector.LinearVector;

public interface GradientOptimizable extends Optimizable {
	public LinearVector getValueGradient();
	public void setRegularizer(Regularizer regularizer);
	
}
