/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;

import java.util.Arrays;
import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * base class for stochastic, gradient-based optimization methods. provides
 * facilities for regularization, gradient truncation
 * 
 * @author jattenberg
 * 
 */
public abstract class AbstractGradientStochasticOptimizer implements
		GradientStochasticOptimizer {

	protected final int dimension;
	protected final boolean bias;
	protected final AnnealingSchedule annealingSchedule;

	protected final GradientTruncation truncation;
	protected final Map<LinearCoefficientLossType, Double> coefficientWeights;
	protected final double regularizationWeight;
	protected final boolean regularizeIntercept;

	protected int epoch;
	protected transient int[] lastAccessed;

	// combined effect multiplier on
	// all regularizations

	protected AbstractGradientStochasticOptimizer(int dimension, boolean bias,
			AnnealingSchedule annealingSchedule, GradientTruncation truncation,
			Map<LinearCoefficientLossType, Double> coefficientWeights,
			boolean regularizeIntercept, double regularizationWeight) {

		checkArgument(
				dimension > 0 && !Double.isInfinite(dimension)
						&& !Double.isNaN(dimension),
				"dimension must be positive and < infinity, given: %s",
				dimension);
		this.dimension = dimension;
		this.bias = bias;

		this.annealingSchedule = annealingSchedule;

		this.truncation = truncation;

		this.coefficientWeights = coefficientWeights;
		this.regularizationWeight = regularizationWeight;
		this.regularizeIntercept = regularizeIntercept;

		epoch = 0;
		lastAccessed = new int[dimension];
		Arrays.fill(lastAccessed, epoch - 1);

	}

	@Override
	public Optimizable update(Optimizable function) {
		double loss = function.computeLoss();
		Gradient gradient = function.computeGradient();
		LinearVector parameters = function.getVector();
		if (annealingSchedule.considerLoss(epoch, loss)) {
			LinearVector regularizationVector = considerRegularizationLoss(
					parameters, gradient);
			// have gradient and regularization gradient, incoporate into the
			// model
			updateModel(function, regularizationVector);
			epoch++;
		}
		truncate(function);
		return function;
	}

	@Override
	public Optimizable cleanup(Optimizable function) {
		LinearVector regularizationGradient = LinearVectorFactory
				.getVector(dimension);
		LinearVector parameters = function.getVector();
		for (int i = 0; i < dimension; i++)
			regularizationGradient.resetValue(i,
					regularizationLoss(parameters, i));
		updateModel(function, regularizationGradient);
		truncate(function);
		epoch++;
		function.setParameters(parameters);
		return function;

	}

	protected void truncate(Optimizable function) {
		truncation.truncateParameters(function.getVector());
	}

	protected LinearVector considerRegularizationLoss(LinearVector parameters,
			LinearVector gradient) {
		LinearVector regularizationVector = LinearVectorFactory
				.getVector(dimension);
		for (int x_i = 0; x_i < dimension; x_i++) {
			regularizationVector.resetValue(x_i,
					regularizationLoss(parameters, x_i));
		}
		return regularizationVector;
	}

	/**
	 * returns the gradient of the regularization loss along a particular
	 * dimension
	 * 
	 * @param dim
	 *            dimension of the model being regularized
	 * @return
	 */
	protected double regularizationLoss(LinearVector parameters, int dim) {

		if ((dim == dimension - 1 && bias && !regularizeIntercept)
				|| !(regularizationWeight > 0))
			return 0;
		double parameterValue = parameters.getValue(dim);
		double loss = 0;

		for (LinearCoefficientLossType lossType : coefficientWeights.keySet()) {
			if (coefficientWeights.get(lossType) > 0)
				loss += lossType.gradient(parameterValue,
						coefficientWeights.get(lossType));
		}

		loss *= (regularizationWeight * (epoch - lastAccessed[dim]))
				/ dimension;
		lastAccessed[dim] = epoch;

		return loss;
	}

	public AnnealingSchedule getAnnealingSchedule() {
		return annealingSchedule;
	}

	public GradientTruncation getTruncation() {
		return truncation;
	}

	public Map<LinearCoefficientLossType, Double> getCoefficientWeights() {
		return coefficientWeights;
	}

	public boolean shouldRegularizeIntercept() {
		return regularizeIntercept;
	}

	public int getEpoch() {
		return epoch;
	}

	public double getRegularizationWeight() {
		return this.regularizationWeight;
	}

	public int getDimension() {
		return dimension;
	}

	public boolean isBias() {
		return bias;
	}

	protected abstract Optimizable updateModel(Optimizable function,
			LinearVector regularizationGradient);

	protected abstract Optimizable updateModelWithRegularization(
			Optimizable function, LinearVector regularizationGradient);

}
