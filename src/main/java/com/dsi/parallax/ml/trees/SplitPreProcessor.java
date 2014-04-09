/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.Target;

/**
 * Interface for systems that pre-process instances before each split <br>
 * example uses may be to down sample a large group of instances, resample to
 * class balance, etc.
 */
public interface SplitPreProcessor {

	/**
	 * Preprocess the input set of instances before splitting
	 * 
	 * @param <T>
	 *            the type of {@link Target} considered
	 * @param <I>
	 *            the type of instances used as training data
	 * @param instances
	 *            the training data
	 * @return the preprocessed training data
	 */
	public <T extends Target, I extends Instances<? extends Instance<T>>> I preprocess(
			I instances);
}
