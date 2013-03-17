/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.projection.Projection;
import com.parallax.ml.target.Target;
import com.parallax.ml.util.pair.GenericPair;


// TODO: Auto-generated Javadoc
/**
 * The Class TreeBuilder.
 *
 * @param <T> the generic type
 */
public class TreeBuilder<T extends Target> {

	/** The tree terminator. */
	protected List<Terminator<T>> treeTerminator;

	/** The pruner. */
	protected Pruner<T> pruner;

	/** The instances. */
	protected Instances<? extends Instance<T>> instances;

	/** The leaf creator. */
	protected LeafCreator<T> leafCreator;

	/** The preprocessor. */
	protected SplitPreProcessor preprocessor;

	/** The attribute value cache. */
	protected AttributeValueCache attributeValueCache;

	/** The splitter. */
	protected Splitter<T> splitter;

	/** The attempts. */
	protected int attempts = 10;

	/** The use pre pruning. */
	protected boolean usePrePruning = true;

	/** The split criterion. */
	protected SplitCriterion<T> splitCriterion;

	/** The projection factory. */
	protected ProjectionFactory projectionFactory;

	/** The projected size. */
	protected double projectedSize = 0.75;

	/**
	 * Instantiates a new tree builder.
	 *
	 * @param instances the instances
	 */
	public TreeBuilder(Instances<? extends Instance<T>> instances) {
		this.instances = instances;
		treeTerminator = Lists.newArrayList();
		attributeValueCache = new RecomputingAttributeValueCache();
	}

	/**
	 * Builds the tree.
	 *
	 * @param <I> the generic type
	 * @param instances the instances
	 * @return the root
	 */
	public <I extends Instances<? extends Instance<T>>> Root<T> buildTree(
			I instances) {

		Root<T> root = Root.buildRoot(instances);
		if (shouldStop(instances, 0))
			root.addLeaf(leafCreator.changeTreeToLeaf(root, instances));
		else
			buildTree(root, root, instances, 1);

		if (null != pruner)
			pruner.prune(root);

		cleanup(root);
		root.resetLeaves();
		return root;
	}

	/**
	 * decends the tree, removing any remaining instances in the nodes.
	 *
	 * @param node the node
	 */
	private void cleanup(Tree<T> node) {
		node.cleanUpInstances();
		if (!node.isLeaf()) {
			for (Tree<T> child : node.getChildren())
				cleanup(child);
		}

	}

	/**
	 * Builds the tree.
	 *
	 * @param <I> the generic type
	 * @param root the root
	 * @param currentNode the current node
	 * @param instances the instances
	 * @param depth the depth
	 */
	@SuppressWarnings("unchecked")
	protected <I extends Instances<? extends Instance<T>>> void buildTree(
			Root<T> root, Tree<T> currentNode, I instances, int depth) {
		if (shouldStop(instances, depth)) {
			root.addLeaf(leafCreator.changeTreeToLeaf(currentNode, instances));
			return;
		}

		I training;
		if (null != preprocessor)
			training = preprocessor.preprocess(instances);
		else
			training = instances;

		if (projectionFactory != null) {
			Projection projection = projectionFactory.buildProjection(
					training.getDimensions(), projectedSize);
			currentNode.addProjection(projection);
			training = (I) training.project(projection);
		}

		List<Benefit> benefits = computeBenefits(training); // have list of
															// benefits for
															// attribute splits
		boolean splitFound = false;
		for (int attempt = 0; attempt <= attempts; attempt++) {
			if (benefits.size() == 0)
				break;

			Benefit bestBenefit = benefits.remove(0);

			if (usePrePruning && bestBenefit.getUtility() <= 0)
				continue;

			int splitIndex = bestBenefit.getIndex();
			double splitValue = bestBenefit.getSplit();

			GenericPair<I, I> leftAndRight = (GenericPair<I, I>) training
					.splitOnValue(splitIndex, splitValue);

			boolean splitOk = true;
			if (usePrePruning) {
				if (shouldStop(leftAndRight.first, depth + 1)
						|| shouldStop(leftAndRight.second, depth + 1))
					splitOk = false;
			}

			if (splitOk) {
				// make trees, add them to parent

				Tree<T> leftTree = new Tree<T>(leftAndRight.first,
						new LessThanOrEqualsToSplitCondition(splitIndex,
								splitValue), currentNode);
				Tree<T> rightTree = new Tree<T>(leftAndRight.second,
						new GreaterThanSplitCondition(splitIndex, splitValue),
						currentNode);

				buildTree(root, leftTree, leftAndRight.first, depth + 1);
				buildTree(root, rightTree, leftAndRight.second, depth + 1);

				splitFound = true;
				break;
			}

		}

		if (!splitFound)
			root.addLeaf(leafCreator.changeTreeToLeaf(currentNode, training));
	}

	/**
	 * Compute benefits.
	 *
	 * @param <I> the generic type
	 * @param training the training
	 * @return the list
	 */
	protected <I extends Instances<? extends Instance<T>>> List<Benefit> computeBenefits(
			I training) {
		List<Benefit> benefits = Lists.newArrayList();

		for (int dim = 0; dim < training.getDimensions(); dim++) {
			Benefit benefit = computeBenefit(training, dim);
			if (null != benefit)
				benefits.add(benefit);
		}
		Collections.sort(benefits);
		return benefits;
	}

	/**
	 * Compute benefit.
	 *
	 * @param <I> the generic type
	 * @param training the training
	 * @param dim the dim
	 * @return the benefit
	 */
	protected <I extends Instances<? extends Instance<T>>> Benefit computeBenefit(
			I training, int dim) {
		AttributeValueSet<T> attributeValues = attributeValueCache
				.getAttributeValueSet(training, dim);
		return splitter.buildSplit(splitCriterion, attributeValues, dim);
	}

	/**
	 * Should stop.
	 *
	 * @param <I> the generic type
	 * @param instances the instances
	 * @param depth the depth
	 * @return true, if successful
	 */
	protected <I extends Instances<? extends Instance<T>>> boolean shouldStop(
			I instances, int depth) {
		for (Terminator<T> terminator : treeTerminator) {
			if (terminator.terminate(instances, depth))
				return true;

		}
		return false;
	}

	// methods for adding settings to the tree builder. employ method chaining.

	/**
	 * Adds the terminator.
	 *
	 * @param terminator the terminator
	 * @return the tree builder
	 */
	public TreeBuilder<T> addTerminator(Terminator<T> terminator) {
		treeTerminator.add(terminator);
		return this;
	}

	/**
	 * Sets the pruner.
	 *
	 * @param pruner the pruner
	 * @return the tree builder
	 */
	public TreeBuilder<T> setPruner(Pruner<T> pruner) {
		this.pruner = pruner;
		return this;
	}

	/**
	 * Sets the instances.
	 *
	 * @param instances the instances
	 * @return the tree builder
	 */
	public TreeBuilder<T> setInstances(
			Instances<? extends Instance<T>> instances) {
		this.instances = instances;
		return this;
	}

	/**
	 * Sets the leaf creator.
	 *
	 * @param leafCreator the leaf creator
	 * @return the tree builder
	 */
	public TreeBuilder<T> setLeafCreator(LeafCreator<T> leafCreator) {
		this.leafCreator = leafCreator;
		return this;
	}

	/**
	 * Sets the split pre processor.
	 *
	 * @param preprocessor the preprocessor
	 * @return the tree builder
	 */
	public TreeBuilder<T> setSplitPreProcessor(SplitPreProcessor preprocessor) {
		this.preprocessor = preprocessor;
		return this;
	}

	/**
	 * Sets the attribute value cache.
	 *
	 * @param cache the cache
	 * @return the tree builder
	 */
	public TreeBuilder<T> setAttributeValueCache(AttributeValueCache cache) {
		this.attributeValueCache = cache;
		return this;
	}

	/**
	 * Sets the splitter.
	 *
	 * @param splitter the splitter
	 * @return the tree builder
	 */
	public TreeBuilder<T> setSplitter(Splitter<T> splitter) {
		this.splitter = splitter;
		return this;
	}

	/**
	 * Sets the prepruning attempts.
	 *
	 * @param attempts the attempts
	 * @return the tree builder
	 */
	public TreeBuilder<T> setPrepruningAttempts(int attempts) {
		this.attempts = attempts;
		return this;
	}

	/**
	 * Sets the use pre pruning.
	 *
	 * @param usePrePruning the use pre pruning
	 * @return the tree builder
	 */
	public TreeBuilder<T> setUsePrePruning(boolean usePrePruning) {
		this.usePrePruning = usePrePruning;
		return this;
	}

	/**
	 * Sets the split criterion.
	 *
	 * @param splitCriterion the split criterion
	 * @return the tree builder
	 */
	public TreeBuilder<T> setSplitCriterion(SplitCriterion<T> splitCriterion) {
		this.splitCriterion = splitCriterion;
		return this;
	}

	/**
	 * Sets the projection factory.
	 *
	 * @param projectionFactory the projection factory
	 * @return the tree builder
	 */
	public TreeBuilder<T> setProjectionFactory(
			ProjectionFactory projectionFactory) {
		this.projectionFactory = projectionFactory;
		return this;
	}
	
	/**
	 * Sets the projection ratio.
	 *
	 * @param ratio the ratio
	 * @return the tree builder
	 */
	public TreeBuilder<T> setProjectionRatio(double ratio) {
		checkArgument(ratio > 0., "projection ratio must be positive, given: %s", ratio);
		this.projectedSize = ratio;
		return this;
	}

}
