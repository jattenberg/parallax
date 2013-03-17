/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

/**
 * A terminator that seeks to avoid empty nodes, nodes with no training examples
 * present.
 * 
 * @param <T>
 *            the type of Target being considered.
 */
public class EmptyTerminator<T extends Target> implements Terminator<T> {

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.Terminator#terminate(com.parallax.ml.instance.Instances
	 * , int)
	 */
	@Override
	public <I extends Instances<? extends Instance<T>>> boolean terminate(
			I instances, int depth) {
		return instances.size() == 0;
	}

}
