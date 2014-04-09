/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.projection.HashProjection;
import com.dsi.parallax.ml.projection.Projection;

// TODO: Auto-generated Javadoc
/**
 * A factory for creating RandomProjection objects.
 */
public class RandomProjectionFactory implements ProjectionFactory {

	/** The salt. */
	private int salt;

	/**
	 * Instantiates a new random projection factory.
	 */
	public RandomProjectionFactory() {
		this(12345);
	}

	/**
	 * Instantiates a new random projection factory.
	 *
	 * @param salt the salt
	 */
	public RandomProjectionFactory(int salt) {
		this.salt = salt;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.ProjectionFactory#buildProjection(int, double)
	 */
	@Override
	public Projection buildProjection(int inputDim, double percentage) {
		int outdim = (int) Math.max(1, Math.round(inputDim * percentage));
		return new HashProjection(inputDim, outdim, salt++);
	}
}
