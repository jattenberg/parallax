/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.optimizable;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

/**
 * A classifier that optimizes L2 loss, capable of general gradient based
 * optimization
 */
public class GradientUpdateableL2 extends
		AbstractGradientUpdateableClassifier<GradientUpdateableL2> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3067505636755093391L;

	/**
	 * Instantiates a new gradient updateable l2.
	 * 
	 * @param builder
	 *            used to construct the optimization procedure used for model
	 *            training
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradientUpdateableL2(
			StochasticGradientOptimizationBuilder<?> builder, int dimension,
			boolean bias) {
		super(builder, dimension, bias);
		initialize();
	}

	/**
	 * Instantiates a new gradient updateable l2.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradientUpdateableL2(int dimension, boolean bias) {
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
		return VectorUtils.dotProduct(parameters, inst);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.optimizable.AbstractGradientUpdateableClassifier
	 * #initialize()
	 */
	@Override
	public GradientUpdateableL2 initialize() {
		super.initialize();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected GradientUpdateableL2 getModel() {
		return this;
	}

	@Override
	public Gradient computeGradient(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;
		LinearVector gradVector = LinearVectorFactory.getVector(dimension);
		if (!MLUtils.floatingPointEquals(denominator, 0)) {

			for (BinaryClassificationInstance inst : instances) {
				double prediction = VectorUtils.dotProduct(params, inst);
				double lossPart = Math.pow(prediction
						- inst.getLabel().getValue(), 2);
				loss += lossPart / denominator;

				for (int x_i : inst) {
					double update = -2
							* (prediction - inst.getLabel().getValue())
							* inst.getValue(x_i) / denominator;
					gradVector.updateValue(x_i, -update);
				}
			}
		}
		Gradient grad = new Gradient(gradVector, loss);
		return grad;
	}

	@Override
	public double computeLoss(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;
		for (BinaryClassificationInstance inst : instances) {
			double prediction = VectorUtils.dotProduct(params, inst);
			double lossPart = Math.pow(prediction - inst.getLabel().getValue(),
					2);
			loss += lossPart;
		}
		return loss / denominator;
	}

	@Override
	public String toString() {
		return parameters.toString();
	}
}
