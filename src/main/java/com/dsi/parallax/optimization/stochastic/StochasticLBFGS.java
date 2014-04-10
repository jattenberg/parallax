/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ********************
import com.google.common.collect.Lists;

import com.google.common.collect.Iterables;

import com.google.common.collect.Iterables;
 **********************************************************/
package com.dsi.parallax.optimization.stochastic;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.google.common.collect.Lists;

//TODO: still baking
public class StochasticLBFGS extends AbstractGradientStochasticOptimizer {

	/** cache of s_{t-i} to y_{t-i} for online LBFGS */
	private final LinkedHashMap<LinearVector, LinearVector> stYtCache;
	private final double lambda;
	private final double epsilon;

	public StochasticLBFGS(int dimension, boolean bias,
			AnnealingSchedule annealingSchedule, GradientTruncation truncation,
			Map<LinearCoefficientLossType, Double> coefficientWeights,
			boolean regularizeIntercept, double regularizationWeight,
			int bandwidth, double epsilon, double lambda) {
		super(dimension, bias, annealingSchedule, truncation,
				coefficientWeights, regularizeIntercept, regularizationWeight);
		checkArgument(bandwidth >= 1,
				"bandwidth must be greater than 1, given: %s", bandwidth);
		stYtCache = new Cache(bandwidth);
		checkArgument(lambda >= 0, "lambda must be non-negative given: %s",
				lambda);
		this.lambda = lambda;

		checkArgument(epsilon > 0, "epsilon must be greater than 0. given: %s",
				epsilon);
		this.epsilon = epsilon;
	}

	@Override
	public Optimizable updateModel(Optimizable function,
			LinearVector regularizationGradient) {
		Gradient gradient = function.computeGradient();
		if (annealingSchedule.considerLoss(epoch, gradient.getLoss())) {
			gradient.minusEquals(regularizationGradient);
			LBFGSUpdate(function, gradient);
		}
		return function;
	}

	@Override
	public Optimizable updateModelWithRegularization(Optimizable function,
			LinearVector regularizationGradient) {
		LBFGSUpdate(function, regularizationGradient.timesEquals(-1));
		return function;
	}

	private void LBFGSUpdate(Optimizable function, LinearVector grad) {
		// 1. p_t = -grad_t
		LinearVector p_t = LinearVectorFactory.getScaledVector(grad, -1.);
		epoch++;

		List<LinearVector> stList = Lists.newArrayList(stYtCache.keySet()); // s_ti
																			// in
																			// oldest
																			// to
																			// newest
																			// order
		double[] alphas = new double[stList.size()];
		int ct = 0;

		for (LinearVector s_ti : Lists.reverse(stList)) {
			LinearVector y_ti = stYtCache.get(s_ti);

			double numerator = VectorUtils.dotProduct(s_ti, p_t);
			double denominator = VectorUtils.dotProduct(s_ti, y_ti);
			double alpha = numerator / denominator;
			alphas[ct++] = alpha;
			p_t.minusEqualsVectorTimes(y_ti, alpha);
		}

		if (stYtCache.size() == 0) {
			p_t.timesEquals(epsilon);
		} else {
			double updateFactor = 0.;

			for (LinearVector s_ti : stList) {
				LinearVector y_ti = stYtCache.get(s_ti);

				double numerator = VectorUtils.dotProduct(s_ti, y_ti);
				double denominator = VectorUtils.dotProduct(y_ti, y_ti);
				updateFactor += numerator / denominator;
			}
			p_t.timesEquals(updateFactor / stYtCache.size());
		}

		for (LinearVector s_ti : stList) {
			LinearVector y_ti = stYtCache.get(s_ti);
			double numerator = VectorUtils.dotProduct(y_ti, p_t);
			double denominator = VectorUtils.dotProduct(y_ti, s_ti);

			double beta = numerator / denominator;
			double alpha = alphas[--ct];
			p_t.plusEqualsVectorTimes(s_ti, alpha - beta);
		}
		LinearVector s_t = LinearVectorFactory.getVector(dimension);
		for (int x_i : p_t) {
			s_t.resetValue(x_i, annealingSchedule.learningRate(epoch, x_i)*p_t.getValue(x_i));
		}

		LinearVector parameters = LinearVectorFactory.getVector(function.getVector());
		parameters.plusEquals(s_t);
		
		LinearVector y_t = LinearVectorFactory.getScaledVector(grad, -1);
		y_t.plusEqualsVectorTimes(s_t, lambda);
		y_t.plusEquals(function.computeGradient(parameters));
		function.setParameters(parameters);
		stYtCache.put(s_t, y_t);
	}

	public static class Cache extends LinkedHashMap<LinearVector, LinearVector> {

		private static final long serialVersionUID = 6876090945205389388L;
		private final int capacity;

		public Cache(int capacity) {
			super(capacity + 1, 1.1f, true);
			this.capacity = capacity;
		}

		@Override
		public boolean removeEldestEntry(
				Map.Entry<LinearVector, LinearVector> eldest) {
			return size() > capacity;
		}
	}
}
