/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * takes a node in a decision tree, turns it to a leaf.
 *
 * @param <T> the generic type
 * @author jattenberg
 */
public interface LeafCreator<T extends Target> {

	/**
	 * Change tree to leaf.
	 *
	 * @param node the node
	 * @param instances the instances
	 * @return the tree
	 */
	public Tree<T> changeTreeToLeaf(Tree<T> node, Instances<? extends Instance<T>> instances);
}
