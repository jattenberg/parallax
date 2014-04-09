/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Map;
import java.util.Set;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;

public class StochasaticGradientDescent extends
		AbstractGradientStochasticOptimizer {

	/**
	 * @param dimension
	 * @param bias
	 * @param annealingSchedule
	 * @param truncation
	 * @param coefficientWeights
	 * @param regularizeIntercept
	 * @param regularizationWeight
	 */
	public StochasaticGradientDescent(int dimension, boolean bias,
			AnnealingSchedule annealingSchedule, GradientTruncation truncation,
			Map<LinearCoefficientLossType, Double> coefficientWeights,
			boolean regularizeIntercept, double regularizationWeight) {
		super(dimension, bias, annealingSchedule, truncation,
				coefficientWeights, regularizeIntercept, regularizationWeight);
	}

	@Override
	protected Optimizable updateModel(Optimizable function,
			LinearVector regularizationGradient) {

		LinearVector parameter = function.getVector();
		Gradient gradient = function.computeGradient();

		if (annealingSchedule.considerLoss(epoch, gradient.getLoss())) {
			epoch++;
			checkArgument(
					parameter.size() == gradient.size(),
					"parameter (size %s) and gradient (size %s) should be of the same dimension",
					parameter.size(), gradient.size());
			Set<Integer> union = gradient.getFeatureIndicies();
			union.addAll(regularizationGradient.getFeatureIndicies());

			for (int x_i : union) {
				double parameterValue = parameter.getValue(x_i);
				parameterValue -= annealingSchedule.learningRate(epoch, x_i)
						* gradient.getValue(x_i);

				if (parameterValue > 0)
					parameterValue = Math.max(0, parameterValue
							+ regularizationGradient.getValue(x_i));
				else
					parameterValue = Math.min(0, parameterValue
							+ regularizationGradient.getValue(x_i));
				parameter.resetValue(x_i, parameterValue);
			}
		}
		function.setParameters(parameter);
		return function;

	}

	@Override
	protected Optimizable updateModelWithRegularization(Optimizable function,
			LinearVector regularizationGradient) {
		LinearVector parameter = function.getVector();
		checkArgument(
				parameter.size() == regularizationGradient.size(),
				"parameter (size %s) and gradient (size %s) should be of the same dimension",
				parameter.size(), regularizationGradient.size());

		for (int x_i : regularizationGradient) {
			double parameterValue = parameter.getValue(x_i);
			if (parameterValue > 0)
				parameterValue = Math.max(0, parameterValue
						- regularizationGradient.getValue(x_i));
			else
				parameterValue = Math.min(0, parameterValue
						- regularizationGradient.getValue(x_i));
			parameter.resetValue(x_i, parameterValue);
		}
		function.setParameters(parameter);
		return function;
	}

}
