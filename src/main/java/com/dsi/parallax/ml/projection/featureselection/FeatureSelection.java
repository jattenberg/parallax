/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection.featureselection;

import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.projection.Projection;

/**
 * interface for classes that select some subset of features from instances can
 * be used to perform projection on linear vectors of the same dimension.
 * 
 * @author jattenberg
 */
public interface FeatureSelection extends Projection {

	/**
	 * Trains the data structures required to perform feature selection
	 * 
	 * @param instances
	 *            the training data to be considered
	 */
	public void build(Instances<?> instances);

	/**
	 * Sets the number of input features to retain
	 * 
	 * @param numToKeep
	 *            the new number to keep
	 */
	public void setNumberToKeep(int numToKeep);

	/**
	 * Gets the number of input features to retain
	 * 
	 * @return the number to keep
	 */
	public int getNumberToKeep();

	/**
	 * Gets the indicies of features from the input space that are to be
	 * retained in the output space.
	 * 
	 * @return the indicies of kept features
	 */
	public int[] getKeptFeatures();
}
