/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;

/**
 * A class for creating leaves on decision tree classifiers. Here we replace a
 * sub-tree with a leaf containing a predictive model. We then trains the model,
 * and updates all pointers.
 * 
 * @author jattenberg
 * 
 */
public class ClassifierLeafCreator implements
		LeafCreator<BinaryClassificationTarget> {

	/** The builder. */
	private final ClassifierBuilder<?, ?> builder;

	/**
	 * Instantiates a new classifier leaf creator.
	 * 
	 * @param builder
	 *            a builder of the desired classifier type and configuration.
	 */
	public ClassifierLeafCreator(ClassifierBuilder<?, ?> builder) {
		this.builder = builder;
	}

	/**
	 * takes a tree node and makes it a leaf, setting the internal classifier
	 * and training it.
	 * 
	 * @param node
	 *            the sub-tree to be turned into a leaf
	 * @param instances
	 *            the training data that have filtered back to the sub-tree in
	 *            question
	 * @return the input sub-tree after the appropriate changes have been made.
	 */
	@Override
	public Tree<BinaryClassificationTarget> changeTreeToLeaf(
			Tree<BinaryClassificationTarget> node,
			Instances<? extends Instance<BinaryClassificationTarget>> instances) {
		node.setToLeaf();
		Classifier<?> model = null;
		model = builder.build();
		model.train(instances);

		node.setModel(model);
		return node;
	}

}
