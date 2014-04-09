/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.linesearch;

import com.dsi.parallax.ml.vector.LinearVector;

public interface LineOptimizer {
	/** Returns the last step size used. */
	public double optimize (LinearVector line, double initialStep);
}
