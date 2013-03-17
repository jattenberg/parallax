/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

/**
 * Interface for classes that decide that no further branching should be
 * performed at a particular node in a decision tree construction
 * 
 * @param <T>
 *            the type of label used
 */
public interface Terminator<T extends Target> {

	/**
	 * Terminate, allow no further splitting at a node
	 * 
	 * @param <I>
	 *            the type of instances used
	 * @param instances
	 *            the training data that has filtered to a node
	 * @param depth
	 *            the depth of the current node in the tree
	 * @return true, if no further branching should be performed
	 */
	public <I extends Instances<? extends Instance<T>>> boolean terminate(
			I instances, int depth);
}
