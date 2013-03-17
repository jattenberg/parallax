/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear.optimizable;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.MLUtils;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.ml.vector.util.VectorUtils;
import com.parallax.optimization.Gradient;
import com.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

/**
 * A classifier that optimizes the subgradient of Quadratic Hinge Loss, capable
 * of general gradient-based optimization. <br>
 * See
 * {@link <a href="http://ttic.uchicago.edu/~dmcallester/ttic101-06/lectures/genreg/genreg.pdf">Regularized Regression</a>}
 * and
 * {@link <a href="http://stat.rutgers.edu/home/tzhang/papers/icml04-stograd.pdf">Solving Large Scale Prediction Problems using Stochastic Gradient Descent</a>}
 * for more info
 */
public class GradienUpdateableQuadraticSVM extends
		AbstractGradientUpdateableClassifier<GradienUpdateableQuadraticSVM> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 2366794254995489973L;

	/** The gamma used to tune the quadratic SVM. */
	private double gamma = 2.0;

	/**
	 * Instantiates a new gradient optimizable quadratic svm.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradienUpdateableQuadraticSVM(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/**
	 * Instantiates a new gradient optimizable quadratic svm.
	 * 
	 * @param builder
	 *            the builder
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public GradienUpdateableQuadraticSVM(
			StochasticGradientOptimizationBuilder<?> builder, int dimension,
			boolean bias) {
		super(builder, dimension, bias);
		initialize();
	}

	/**
	 * Q svm loss.
	 * 
	 * @param <I>
	 *            the generic type
	 * @param inst
	 *            the inst
	 * @return the double
	 */
	private <I extends Instance<BinaryClassificationTarget>> double qSVMLoss(
			I inst, LinearVector params) {
		double yp = VectorUtils.dotProduct(params, inst)
				* MLUtils.probToSVMInterval(inst.getLabel().getValue());

		if (yp > 1. - gamma) {
			return (1 / (2. * gamma)) * Math.pow(Math.max(0, 1 - yp), 2);
		} else {
			return 1. - gamma / 2. - yp;
		}
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
		return MLUtils.svmIntervalToProb(VectorUtils.dotProduct(parameters,
				inst));
	}

	/**
	 * Gets the gamma.
	 * 
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma;
	}

	/**
	 * Sets the gamma.
	 * 
	 * @param gamma
	 *            the gamma
	 * @return the gradient optimizable quadratic svm
	 */
	public GradienUpdateableQuadraticSVM setGamma(double gamma) {
		checkArgument(gamma > 0, "gamma must be positive. Given: %s", gamma);
		this.gamma = gamma;
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.optimizable.AbstractGradientUpdateableClassifier
	 * #initialize()
	 */
	@Override
	public GradienUpdateableQuadraticSVM initialize() {
		super.initialize();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected GradienUpdateableQuadraticSVM getModel() {
		return this;
	}

	@Override
	public Gradient computeGradient(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;
		LinearVector gradVector = LinearVectorFactory.getVector(dimension);

		for (BinaryClassificationInstance inst : instances) {

			double iLabel = MLUtils.probToSVMInterval(inst.getLabel()
					.getValue());

			double yp = VectorUtils.dotProduct(params, inst)
					* MLUtils.probToSVMInterval(inst.getLabel().getValue());
			double iLoss;

			if (yp > 1. - gamma) {
				iLoss = (1 / (2. * gamma)) * Math.pow(Math.max(0, 1 - yp), 2);
				if (1 - yp > 0) {
					for (int x_i : inst) {
						double update = -gamma * (1 - yp)
								* inst.getFeatureValue(x_i) * iLabel;
						gradVector.updateValue(x_i, update);
					}
				}
			} else {
				iLoss = 1. - gamma / 2. - yp;
				for (int x_i : inst) {
					double update = -inst.getFeatureValue(x_i) * iLabel;
					gradVector.updateValue(x_i, update);
				}
			}
			loss += iLoss / denominator;
		}

		return new Gradient(gradVector, loss);
	}

	/**
	 * max(0, 1-y*f(x))^2/(2*gamma) when y*f(x) >= 1-gamma, <br>
	 * 
	 * 1-gamma/2 -y*f(x) otherwise
	 */
	@Override
	public double computeLoss(LinearVector params) {
		double denominator = instances.size();
		double loss = 0d;
		for (BinaryClassificationInstance inst : instances) {
			double iLoss = qSVMLoss(inst, params);
			loss += iLoss;
		}
		return loss / denominator;
	}
}
