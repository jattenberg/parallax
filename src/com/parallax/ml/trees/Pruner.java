/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * class for pruning trees. used for removing sub-trees that may have
 * arisen from overfitting or noise.
 *
 * @param <T> the target type used.
 * @author jattenberg
 */
public interface Pruner<T extends Target> {

	/**
	 * Prune.
	 *
	 * @param tree the tree
	 */
	public void prune(Tree<T> tree);
}
