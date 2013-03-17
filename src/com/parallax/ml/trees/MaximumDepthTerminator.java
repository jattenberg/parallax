/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * The Class MaximumDepthTerminator.
 *
 * @param <T> the generic type
 */
public class MaximumDepthTerminator<T extends Target> implements Terminator<T> {

	/** The max depth. */
	private final int maxDepth;

	/**
	 * Instantiates a new maximum depth terminator.
	 *
	 * @param maxDepth the max depth
	 */
	public MaximumDepthTerminator(int maxDepth) {
		this.maxDepth = maxDepth;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.Terminator#terminate(com.parallax.ml.instance.Instances, int)
	 */
	@Override
	public <I extends Instances<? extends Instance<T>>> boolean terminate(I instances,
			int depth) {
		return depth >= maxDepth;
	}

}
