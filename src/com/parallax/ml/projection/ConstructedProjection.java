/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import java.util.Collection;

import com.parallax.ml.vector.LinearVector;

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
