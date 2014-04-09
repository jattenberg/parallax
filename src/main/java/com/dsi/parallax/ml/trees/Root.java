/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import java.util.Map;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.Target;
import com.google.common.collect.Maps;

/**
 * Root node of a decision tree, used to store a map of leaves.
 * 
 * @param <T>
 *            the generic type
 */
public class Root<T extends Target> extends Tree<T> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8959314855543075765L;

	/** integers corresponding to identifiers of the leaves in the tree */
	private Map<Tree<T>, Integer> leafIds;

	/**
	 * Instantiates a new root.
	 * 
	 * @param insts
	 *            - training data supplied to the decision tree trainer
	 */
	public Root(Instances<? extends Instance<T>> insts) {
		super(insts);
		leafIds = Maps.newHashMap();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.trees.Tree#isRoot()
	 */
	@Override
	public boolean isRoot() {
		return true;
	}

	/**
	 * Request the leaf id where a particular instance resolves in the decision
	 * tree.
	 * 
	 * @param inst
	 *            input to be filtered through the decision tree
	 * @return the identifier of the leaf where the input resolves
	 */
	public int idForInstance(Instance<?> inst) {
		if (isLeaf())
			return 0;
		else {
			Tree<T> leaf = decendTree(inst);
			return idForLeaf(leaf);
		}
	}

	/**
	 * Request the identifier for a particular leaf node in the decision tree,
	 * null if the leaf isnt in the tree
	 * 
	 * @param leaf
	 *            where the identifier is being requested
	 * @return the identifier for a particular leaf node in the decision tree,
	 *         null if the leaf isnt in the tree
	 */
	public Integer idForLeaf(Tree<T> leaf) {
		return leafIds.containsKey(leaf) ? leafIds.get(leaf) : null;
	}

	/**
	 * Adds a leaf to the decision tree
	 * 
	 * @param leaf
	 *            the leaf being added
	 */
	public void addLeaf(Tree<T> leaf) {
		leafIds.put(leaf, leafIds.size());
	}

	/**
	 * Reset leaves; decends the tree and reassigns identifiers to the leaf
	 * nodes
	 */
	public void resetLeaves() {
		leafIds = Maps.newHashMap();
		resetLeaves(this);

	}

	/**
	 * Reset leaves; decends the tree and reassigns identifiers to the leaf
	 * nodes
	 * 
	 * @param tree
	 *            the subtree to be decended
	 */
	private void resetLeaves(Tree<T> tree) {
		if (tree.isLeaf())
			addLeaf(tree);
		else {
			for (Tree<T> children : tree.getChildren())
				resetLeaves(children);
		}

	}

	/**
	 * Builds a root node with the supplied instances
	 * 
	 * @param <T>
	 *            the type of label considered
	 * @param <I>
	 *            the type of instance used as training data
	 * @param instances
	 *            the training data
	 * @return the root constructed with the supplied training data
	 */
	public static <T extends Target, I extends Instances<? extends Instance<T>>> Root<T> buildRoot(
			I instances) {
		return new Root<T>(instances);
	}
}
