/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PassiveAggressiveBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/*
 * 
 * also provides techniques for training w and representing x 
 * 
 * note: this maintains a SINGLE classifier for binary classification. multi-class/etc must be implemented externally\
 * by combining several PassiveAggressive classes
 * 
 * See Online Passive Aggressive Algorithms {@link http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf}
 * 
 */
/**
 * The Class PassiveAggressive.
 */
public class PassiveAggressive extends
		AbstractLinearUpdateableClassifier<PassiveAggressive> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5029003140250052678L;

	/** The aggressiveness. */
	private double aggressiveness = 5.;

	/**
	 * Instantiates a new passive aggressive.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @param dense
	 *            the dense
	 */
	public PassiveAggressive(int dimension, boolean bias, boolean dense) {
		super(dimension, bias, 0.0, dense);
	}

	/**
	 * Instantiates a new passive aggressive.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public PassiveAggressive(int dimension, boolean bias) {
		super(dimension, bias, 0.0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.linearupdateable.
	 * AbstractLinearUpdateableClassifier
	 * #updateLinearModel(com.parallax.ml.instance.Instanze)
	 */
	@Override
	protected <I extends Instance<BinaryClassificationTarget>> Gradient computeUpdateGradient(
			I x) {

		double prediction = innerProduct(x);
		double actual = MLUtils.probToSVMInterval(x.getLabel().getValue());

		LinearVector gradientVector = LinearVectorFactory.getVector(dimension);

		double loss = Math.max(0.0, 1d - actual * prediction);
		double tau;
		if (loss == 0.0) {
			return new Gradient(gradientVector);
		}
		double norm = Math.pow(x.L2Norm(), 2d) + (bias ? 1.0 : 0);
		if (MLUtils.floatingPointEquals(0, norm)) {
			return new Gradient(gradientVector);
		}
		tau = Math.min(aggressiveness, loss / norm);

		for (int x_i : x) {
			double y_i = x.getFeatureValue(x_i);
			double update = tau * actual * y_i * x.getWeight();
			gradientVector.updateValue(x_i, update);
		}
		if (bias) {
			double update = tau * actual;
			gradientVector.updateValue(dimension - 1, update);
		}
		return new Gradient(gradientVector);
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.linearupdateable.
	 * AbstractLinearUpdateableClassifier
	 * #innerProduct(com.parallax.ml.instance.Instanze)
	 */
	@Override
	protected double innerProduct(Instance<?> x) {
		return super.innerProduct(x);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public PassiveAggressive initialize() {
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected PassiveAggressive getModel() {
		return this;
	}

	/**
	 * Sets the aggressiveness.
	 * 
	 * @param aggressiveness
	 *            the aggressiveness
	 * @return the passive aggressive
	 */
	public PassiveAggressive setAggressiveness(double aggressiveness) {
		checkArgument(aggressiveness >= 0,
				"aggressiveness must be > 0. input: %f", aggressiveness);
		this.aggressiveness = aggressiveness;
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public double regress(Instance<?> instance) {
		return innerProduct(instance);
	}

	/**
	 * Gets the aggressiveness.
	 * 
	 * @return the aggressiveness
	 */
	public double getAggressiveness() {
		return aggressiveness;
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the passive aggressive builder
	 */
	public static PassiveAggressiveBuilder builder(int dimension, boolean bias) {
		return new PassiveAggressiveBuilder(dimension, bias);
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
		ClassifierEvaluation.evaluate(args, PassiveAggressive.class);
	}

}
