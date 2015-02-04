/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.Target;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * The Class MinExamplesTerminator.
 *
 * @param <T> the generic type
 */
public class MinExamplesTerminator<T extends Target> implements Terminator<T> {

	/** The min examples. */
	private final int minExamples;

	/**
	 * Instantiates a new min examples terminator.
	 *
	 * @param minExamples the min examples
	 */
	public MinExamplesTerminator(int minExamples) {
		checkArgument(minExamples >= 0,
				"minExamples must be positive. given: %s", minExamples);
		this.minExamples = minExamples;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.Terminator#terminate(com.parallax.ml.instance.Instances, int)
	 */
	@Override
	public <I extends Instances<? extends Instance<T>>> boolean terminate(
			I instances, int depth) {
		return instances.size() < minExamples;
	}
}
