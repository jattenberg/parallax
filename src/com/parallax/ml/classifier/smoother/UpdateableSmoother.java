/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * The Interface UpdateableSmoother.
 */
public interface UpdateableSmoother<U extends UpdateableSmoother<U>>
		extends Smoother<U> {

	/**
	 * Update the model based on a new observation of prediction, label pairs.
	 * 
	 * @param p
	 *            the prediction / label pair used to train the reagularizer
	 */
	public void update(PrimitivePair p);
}
