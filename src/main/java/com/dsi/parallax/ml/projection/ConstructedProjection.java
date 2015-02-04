/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import com.dsi.parallax.ml.vector.LinearVector;

import java.util.Collection;

/**
 * interface for projections that require data for construction eg. PCA
 * 
 * @author jattenberg
 * 
 */
public interface ConstructedProjection extends Projection {

	/**
	 * Builds the.
	 * 
	 * @param X
	 *            the x
	 */
	public void build(final Collection<LinearVector> X);

	/**
	 * Checks if is built.
	 * 
	 * @return true, if is built
	 */
	public boolean isBuilt();
}
