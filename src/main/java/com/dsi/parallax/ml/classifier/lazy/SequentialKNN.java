/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelSGD;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.KDTree.Entry;

import java.util.List;

// TODO: Auto-generated Javadoc
/**
 * a flexible implementation of a KNN model TODO: distance weighting.
 * 
 * @author jattenberg
 */
public class SequentialKNN extends
		AbstractUpdateableKDTreeClassifier<SequentialKNN> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -6661450795055899046L;

	/**
	 * Instantiates a new sequential knn.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public SequentialKNN(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	protected double regress(Instance<?> inst) {
		List<Entry<BinaryClassificationTarget>> neighborLabels = findNeighbors(inst);
		if (neighborLabels.size() == 0)
			return 0.5;
		return mixing.computeScore(neighborLabels);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected SequentialKNN getModel() {
		return this;
	}

	/**
	 * The main method.
	 * 
	 * @param args
	 *            the arguments
	 * @throws Exception
	 *             the exception
	 */
	public static void main(String[] args) throws Exception {
		ClassifierEvaluation.evaluate(args, KernelSGD.class);
	}

}
