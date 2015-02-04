/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.trees;

import com.dsi.parallax.ml.classifier.AbstractClassifier;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.trees.*;
import com.dsi.parallax.ml.trees.Terminator;
import com.google.common.collect.Lists;

import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Base class for decision tree-based classifiers. Reduces the specification of
 * most decision tree models to the specification of a few key types:
 * 
 * {@link #buildAdditionalTerminators()} {@link #buildAttributeValueCache()}
 * {@link #buildCriterion()} {@link #buildLeafCreator()}
 * {@link #buildProjectionFactory()} {@link #buildProjectionRatio()}
 * 
 * pretty easy to do something incredibly powerful!
 * 
 * @param <C>
 *            concrete type of tree classifier. Used for method chaining.
 * @author jattenberg
 */
public abstract class AbstractTreeClassifier<C extends AbstractTreeClassifier<C>>
		extends AbstractClassifier<C> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 2736737858468719244L;

	/** The maximum allowable depth for a decision tree. */
	protected int maxDepth = Integer.MAX_VALUE;
	/**
	 * the minimum permissible number of examples in a node for further
	 * splitting
	 */
	protected int minExamples = 0;
	/**
	 * the number of attempts at pruning BEFORE a split is performed. If no
	 * successes are recorded after this many tries, this branch is terminated.
	 */
	protected int prepruningAttempts = 20;

	/**
	 * The minimum allowable entropy in the labels of a node allowable for
	 * further splits
	 */
	protected double minEntropy = 0d;
	/**
	 * if a nested projector is used, the ratio of projection used at each node.
	 */
	protected double projectionRatio = 1.;

	/**
	 * Terminator used for all trees, by default, uses the highest integer value
	 * as a maximum depth
	 */
	protected MaximumDepthTerminator<BinaryClassificationTarget> maxDepthTerminator;

	/**
	 * Root node of the tree data structure. Note that decision trees are
	 * currently implemented as a multibly linked list.
	 */
	protected Root<BinaryClassificationTarget> root;

	/**
	 * Instantiates a new abstract tree classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public AbstractTreeClassifier(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#train(com.parallax.ml.instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances) {
		TreeBuilder<BinaryClassificationTarget> builder = buildTreeBuilder(instances);
		root = builder.buildTree(instances);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	protected double regress(Instance<?> instance) {
		Tree<BinaryClassificationTarget> child = root;

		while (!child.isLeaf()) {
			child = child.decendTree(instance);
		}
		return child.predict(instance).getValue();
	}

	/**
	 * build list of terminators that will be used for halting. EmptyTerminator
	 * is always used. configured by the numerical settings used.
	 * 
	 * @return the list of terminators used.
	 */
	protected List<Terminator<BinaryClassificationTarget>> buildTerminatorList() {
		List<Terminator<BinaryClassificationTarget>> out = Lists.newArrayList();

		out.add(new EmptyTerminator<BinaryClassificationTarget>());
		out.add(new SingleLabelTerminator<BinaryClassificationTarget>());

		if (maxDepth != Integer.MAX_VALUE) {
			out.add(new MaximumDepthTerminator<BinaryClassificationTarget>(
					maxDepth));
		}
		if (minExamples > 0) {
			out.add(new MinExamplesTerminator<BinaryClassificationTarget>(
					minExamples));
		}
		if (minEntropy > 0) {
			out.add(new BinaryTargetEntropyTerminator(minEntropy));
		}
		List<Terminator<BinaryClassificationTarget>> additional = buildAdditionalTerminators();
		if (null != additional) {
			for (Terminator<BinaryClassificationTarget> term : additional) {
				out.add(term);
			}
		}

		return out;
	}

	/**
	 * Builds the tree builder used for constructing the internal decision tree
	 * data structure
	 * 
	 * @param <I>
	 *            type of instances used for training the decision tree model
	 * @param instances
	 *            training data used in model construction
	 * @return the tree builder used for constructing the internal decision tree
	 *         data structure
	 */
	protected <I extends Instances<? extends Instance<BinaryClassificationTarget>>> TreeBuilder<BinaryClassificationTarget> buildTreeBuilder(
			I instances) {
		TreeBuilder<BinaryClassificationTarget> treeBuilder = new TreeBuilder<BinaryClassificationTarget>(
				instances);
		for (Terminator<BinaryClassificationTarget> terminator : buildTerminatorList()) {
			treeBuilder.addTerminator(terminator);
		}
		treeBuilder.setPrepruningAttempts(prepruningAttempts)
				.setSplitter(buildSplitter())
				.setSplitCriterion(buildCriterion())
				.setAttributeValueCache(buildAttributeValueCache())
				.setLeafCreator(buildLeafCreator()).setPruner(buildPruner())
				.setProjectionFactory(buildProjectionFactory())
				.setProjectionRatio(projectionRatio);
		return treeBuilder;
	}

	/**
	 * Generate the class that defines the value attributed to a particular
	 * split in the decision tree.
	 * 
	 * @return the split criterion
	 */
	protected abstract SplitCriterion<BinaryClassificationTarget> buildCriterion();

	/**
	 * Builds the class that actually performs splitting at the nodes.
	 * 
	 * @return the splitter
	 */
	protected abstract Splitter<BinaryClassificationTarget> buildSplitter();

	/**
	 * Builds a cache that stores pre-computed label distributions at different
	 * splits
	 * 
	 * @return the attribute value cache
	 */
	protected abstract AttributeValueCache buildAttributeValueCache();

	/**
	 * Builds a class for creating leaves in the decision tree. Often these will
	 * define a model used for predicting in the leaves. In the simplest case
	 * this will just be a mean or mode label, but can also be something more
	 * complex
	 * 
	 * @return the leaf creator
	 */
	protected abstract LeafCreator<BinaryClassificationTarget> buildLeafCreator();

	/**
	 * Builds the pruner used for reducing complexity in a decision tree model.
	 * 
	 * @return the pruner
	 */
	protected abstract Pruner<BinaryClassificationTarget> buildPruner();

	/**
	 * Builds the projection factory- projections can be inserted into nodes.
	 * Examples can be projected into alternative spaces as they are passed down
	 * the decision tree.
	 * 
	 * @return the projection factory
	 */
	protected abstract ProjectionFactory buildProjectionFactory();

	/**
	 * Returns the projection ratio; the amount of projection that is performed
	 * on instances as they pass through each node.
	 * 
	 * @return the double
	 */
	protected abstract double buildProjectionRatio();

	/**
	 * Define a list of additional terminators that can be used to stop
	 * additional splitting and generate a leaf when building a decision tree.
	 * 
	 * @return the list of additional terminators desired.
	 */
	protected abstract List<Terminator<BinaryClassificationTarget>> buildAdditionalTerminators();

	/**
	 * Sets the max depth permissible in the decision tree. must be
	 * non-negative.
	 * 
	 * @param maxDepth
	 *            the max depth permissible.
	 * @return the model itself used for method chaining
	 */
	public C setMaxDepth(int maxDepth) {
		checkArgument(maxDepth >= 0, "maxDepth must be positive, given %s",
				maxDepth);
		this.maxDepth = maxDepth;
		return model;
	}

	/**
	 * Sets the minimum permissible number of examples in a node for further
	 * splitting
	 * 
	 * @param minExamples
	 *            the min examples
	 * @return the model itself used for method chaining
	 */
	public C setMinExamples(int minExamples) {
		checkArgument(minExamples >= 0,
				"minExamples must be positive, given %s", minExamples);
		this.minExamples = minExamples;
		return model;
	}

	/**
	 * Sets the minimum allowable entropy in the labels of a node allowable for
	 * further splits
	 * 
	 * @param minEntropy
	 *            the min entropy
	 * @return the model itself used for method chaining
	 */
	public C setMinEntropy(double minEntropy) {
		checkArgument(minEntropy >= 0,
				"entropy must be non-negative. given: %s", minEntropy);
		this.minEntropy = minEntropy;
		return model;
	}

	/**
	 * Sets the number of attempts at pruning BEFORE a split is performed. If no
	 * successes are recorded after this many tries, this branch is terminated.
	 * 
	 * @param prepruningAttempts
	 *            the prepruning attempts
	 * @return the model itself used for method chaining
	 */
	public C setPrepruningAttempts(int prepruningAttempts) {
		checkArgument(prepruningAttempts >= 0,
				"prepruningAttempts must be greater than 0, given %s",
				prepruningAttempts);
		this.prepruningAttempts = prepruningAttempts;
		return model;
	}

	/**
	 * Sets the projection ratio. if a nested projector is used, the ratio of
	 * projection used at each node.
	 * 
	 * @param projectionRatio
	 *            the projection ratio
	 * @return the model itself used for method chaining
	 */
	public C setProjectionRatio(double projectionRatio) {
		checkArgument(projectionRatio > 0,
				"projectionRatio must be greater than 0, given %s",
				projectionRatio);
		this.projectionRatio = projectionRatio;
		return model;
	}

}
