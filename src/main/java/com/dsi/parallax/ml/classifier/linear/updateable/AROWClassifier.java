/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.AROWClassifierBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;

import java.util.Arrays;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * linear classifier implementing Adaptive Regularization of Weight Vectors
 * Crammer, Kulesza and Dredze
 * 
 * {@link <a href="http://www.cis.upenn.edu/~kulesza/pubs/arow_nips09.pdf">paper</a>}
 * 
 * a confidence-weighted update for non-seperable data.
 * 
 * @author josh
 * 
 */
public class AROWClassifier extends
		AbstractLinearUpdateableClassifier<AROWClassifier> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -416821294461111306L;

	/** The diag covariance. */
	private double diagCovariance[];

	/** The r. */
	private double r = 0.000001;

	private static final AnnealingSchedule SCHEDULE = new ConstantAnnealingSchedule(
			1d);

	/**
	 * Instantiates a new aROW classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @param dense
	 *            sparse or dense representation
	 */
	public AROWClassifier(int dimension, boolean bias, boolean dense) {
		super(dimension, bias, dense);
		initialize();
	}

	/**
	 * Instantiates a new aROW classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public AROWClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/**
	 * Gets the margin.
	 * 
	 * @param x
	 *            the x
	 * @return the margin
	 */
	private double getMargin(Instance<?> x) {
		double res = 0.0;
		for (int x_i : x)
			res += getParam(x_i) * x.getFeatureValue(x_i);
		return res;
	}

	/**
	 * Gets the confidence.
	 * 
	 * @param x
	 *            the x
	 * @return the confidence
	 */
	private double getConfidence(Instance<?> x) {
		double res = 0.0;
		for (int x_i : x)
			res += diagCovariance[x_i] * Math.pow(x.getFeatureValue(x_i), 2);
		return res;
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
		double margin = getMargin(x);
		double label = MLUtils.probToSVMInterval(x.getLabel().getValue());
		LinearVector gradientVector = LinearVectorFactory.getVector(dimension);

		if (margin * label >= 1)
			return new Gradient(gradientVector);

		double confidence = getConfidence(x);

		if (!MLUtils.floatingPointEquals(confidence + r, 0.)) {
			double beta = 1.0 / (confidence + r);
			double alpha = (1.0 - label * margin) * beta;

			// Update mean
			for (int x_i : x) {
				double update = alpha * label * diagCovariance[x_i]
						* x.getFeatureValue(x_i);
				gradientVector.updateValue(x_i, update);
			}
			if (bias) {
				double update = alpha * label * diagCovariance[dimension - 1];
				gradientVector.updateValue(dimension - 1, update);
			}

			// Update covariance
			for (int x_i : x) {
				diagCovariance[x_i] = 1. / ((1. / diagCovariance[x_i]) + Math
						.pow(x.getFeatureValue(x_i), 2) / r);
			}
			if (bias) {
				diagCovariance[dimension - 1] = 1. / ((1. / diagCovariance[dimension - 1]) + 1. / r);
			}
		}
		return new Gradient(gradientVector);
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
		return getMargin(instance);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public AROWClassifier initialize() {
		initW();
		diagCovariance = new double[dimension];
		Arrays.fill(diagCovariance, 1.);
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected AROWClassifier getModel() {
		return this;
	}

	/**
	 * Sets the R parameter for the AROW algorithm. smaller values yield more
	 * aggressive updates input values must be positive.
	 * 
	 * @param r
	 *            the r value to be considered
	 * @return the AROW classifier itself used for method chaining.
	 */
	public AROWClassifier setR(double r) {
		checkArgument(r >= 0, "r must be > 0 input: %s", r);
		this.r = r;
		return model;
	}

	/**
	 * Gets the R value specified in the current implementation
	 * 
	 * @return the value of r in the current model.
	 */
	public double getR() {
		return r;
	}

	@Override
	public AnnealingSchedule getAnnealingSchedule() {
		return SCHEDULE;
	}

	/**
	 * Gets an AROWBuilder with the specified dimension and bias value.
	 * 
	 * @param dimension
	 *            the dimension of the specified model
	 * @param bias
	 *            should the builder consider a bias term
	 * @return A builder for AROW classifier models
	 */
	public AROWClassifierBuilder builder(int dimension, boolean bias) {
		return new AROWClassifierBuilder(dimension, bias);
	}

	/**
	 * The test main method used for simple use cases for classifiers.
	 * 
	 * @See {@link ClassifierEvaluation}
	 * 
	 * @param the
	 *            command line arguments input at the command line
	 * @throws Exceptions
	 *             due to bad file reads.
	 */
	public static void main(String[] args) throws Exception {
		ClassifierEvaluation.evaluate(args, AROWClassifier.class);
	}

}
