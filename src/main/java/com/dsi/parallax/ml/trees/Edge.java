/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.target.Target;
import com.google.common.collect.ComparisonChain;

import java.io.Serializable;

/**
 * An edge in a decision tree; the connection split condition determining which
 * branch to explore and the next node along that path
 * 
 * @param <T>
 *            the type of Target to be considered in the decision tree
 */
public class Edge<T extends Target> implements Serializable,
		Comparable<Edge<T>> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3158878715541747335L;

	/** The tree at the end of the edge. */
	private final Tree<T> tree;

	/** The condition split condition governing a branch. */
	private final SplitCondition condition;

	/**
	 * Instantiates a new edge.
	 * 
	 * @param tree
	 *            The node at the end of the edge.
	 * @param condition
	 *            The condition split condition governing a branch.
	 */
	public Edge(Tree<T> tree, SplitCondition condition) {
		this.tree = tree;
		this.condition = condition;
	}

	/**
	 * Gets the child.
	 * 
	 * @return the child
	 */
	public Tree<T> getChild() {
		return tree;
	}

	/**
	 * Gets the split condition.
	 * 
	 * @return the split condition
	 */
	public SplitCondition getSplitCondition() {
		return condition;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(Edge<T> e) {
		return ComparisonChain.start()
				.compare(condition.index, e.condition.index)
				.compare(condition.splitValue, e.condition.splitValue)
				.compare(condition.naturalOrder(), e.condition.naturalOrder())
				.result();
	}

}
