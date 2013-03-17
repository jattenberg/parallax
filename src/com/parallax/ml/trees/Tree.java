/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.common.collect.Sets;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.model.Model;
import com.parallax.ml.projection.Projection;
import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * class representing nodes in trees can be a leaf (no children).
 *
 * @param <T> the generic type
 * @author jattenberg
 */
public class Tree<T extends Target> implements Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 451644753659915095L;

	/** The children. */
	private Set<Edge<T>> children;

	/** The parent. */
	private Tree<T> parent;

	/** The insts. */
	protected Instances<? extends Instance<T>> insts = null;

	/** The model. */
	protected Model<T, ?> model;

	/** The projection. */
	protected Projection projection;

	/**
	 * Instantiates a new tree.
	 *
	 * @param insts the insts
	 */
	public Tree(Instances<? extends Instance<T>> insts) {
		children = Sets.newTreeSet();
		this.insts = insts;
	}

	/**
	 * Instantiates a new tree.
	 *
	 * @param insts the insts
	 * @param splitCondition the split condition
	 * @param parent the parent
	 */
	public Tree(Instances<? extends Instance<T>> insts,
			SplitCondition splitCondition, Tree<T> parent) {
		this(insts);
		this.parent = parent;
		parent.addChild(splitCondition, this);
	}

	/**
	 * Checks if is leaf.
	 *
	 * @return true, if is leaf
	 */
	public boolean isLeaf() {
		return children == null || children.size() == 0;
	}

	/**
	 * Checks if is root.
	 *
	 * @return true, if is root
	 */
	public boolean isRoot() {
		return parent == null;
	}

	/**
	 * Gets the instances.
	 *
	 * @return the instances
	 */
	public Instances<? extends Instance<T>> getInstances() {
		return insts;
	}

	/**
	 * Clean up instances.
	 */
	public void cleanUpInstances() {
		insts = null;
	}

	/**
	 * get the set of all children of a node. order is not preserved
	 *
	 * @return the children
	 */
	public Set<Tree<T>> getChildren() {

		return Sets.newHashSet(Iterators.transform(children.iterator(),
				new Function<Edge<T>, Tree<T>>() {

					@Override
					public Tree<T> apply(Edge<T> edge) {
						return edge.getChild();
					}
				}));
	}

	/**
	 * Decend tree.
	 *
	 * @param instance the instance
	 * @return the tree
	 */
	public Tree<T> decendTree(Instance<?> instance) {
		if (isLeaf())
			return null;
		Instance<?> splitInstance = projection == null ? instance : projection
				.project(instance);
		for (Edge<T> edge : children)
			if (edge.getSplitCondition().satisfiesSplit(splitInstance))
				return edge.getChild();
		return null;
	}

	/**
	 * Predict.
	 *
	 * @param instance the instance
	 * @return the t
	 */
	public T predict(Instance<?> instance) {
		if (!isLeaf())
			return null;
		Instance<?> predictInstance = projection == null ? instance
				: projection.project(instance);
		return model.predict(predictInstance);
	}

	/**
	 * Gets the projection.
	 *
	 * @return the projection
	 */
	public Projection getProjection() {
		return projection;
	}

	/**
	 * Sets the model.
	 *
	 * @param model the model
	 * @return the tree
	 */
	public Tree<T> setModel(Model<T, ?> model) {
		this.model = model;
		return this;
	}

	/**
	 * Adds the child.
	 *
	 * @param splitCondition the split condition
	 * @param tree the tree
	 * @return the tree
	 */
	public Tree<T> addChild(SplitCondition splitCondition, Tree<T> tree) {
		return addChild(new Edge<T>(tree, splitCondition));
	}

	/**
	 * Adds the child.
	 *
	 * @param edge the edge
	 * @return the tree
	 */
	public Tree<T> addChild(Edge<T> edge) {
		children.add(edge);
		return this;
	}

	/**
	 * Adds the projection.
	 *
	 * @param projection the projection
	 * @return the tree
	 */
	public Tree<T> addProjection(Projection projection) {
		this.projection = projection;
		return this;
	}

	/**
	 * Sets the to leaf.
	 */
	@SuppressWarnings("unchecked")
	public void setToLeaf() {
		children = Collections.EMPTY_SET;
	}

}
