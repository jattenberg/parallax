/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.projection.Projection;

// TODO: Auto-generated Javadoc
/**
 * A factory for creating Projection objects.
 */
public interface ProjectionFactory {

	/**
	 * Builds the projection.
	 *
	 * @param inputDim the input dim
	 * @param percentage the percentage
	 * @return the projection
	 */
	public Projection buildProjection(int inputDim, double percentage);
}
