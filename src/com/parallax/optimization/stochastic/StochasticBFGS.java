/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;

import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.ml.vector.util.EJMLMatrixDecorator;
import com.parallax.optimization.Gradient;
import com.parallax.optimization.Optimizable;
import com.parallax.optimization.regularization.GradientTruncation;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.optimization.stochastic.anneal.AnnealingSchedule;

public class StochasticBFGS extends AbstractGradientStochasticOptimizer {

	private final double c, lambda;

	private SimpleMatrix B;

	public StochasticBFGS(int dimension, boolean bias,
			AnnealingSchedule annealingSchedule, GradientTruncation truncation,
			Map<LinearCoefficientLossType, Double> coefficientWeights,
			boolean regularizeIntercept, double regularizationWeight,
			double epsilon, double c, double lambda) {
		super(dimension, bias, annealingSchedule, truncation,
				coefficientWeights, regularizeIntercept, regularizationWeight);

		checkArgument(c > 0 && c <= 1, "c must be in (0, 1]. given: %s", c);
		this.c = c;

		checkArgument(lambda >= 0, "lambda must be non-negative given: %s",
				lambda);
		this.lambda = lambda;

		checkArgument(epsilon > 0, "epsilon must be greater than 0. given: %s",
				epsilon);

		double[] diag = new double[dimension];
		Arrays.fill(diag, epsilon);
		B = SimpleMatrix.diag(diag);
	}

	@Override
	protected Optimizable updateModel(Optimizable function,
			LinearVector regularizationGradient) {
		Gradient gradient = function.computeGradient();
		if (annealingSchedule.considerLoss(epoch, gradient.getLoss())) {
			SimpleMatrix grad = new SimpleMatrix(new EJMLMatrixDecorator(
					gradient));
			grad = grad.minus(new SimpleMatrix(new EJMLMatrixDecorator(
					regularizationGradient)));
			BFGSUpdate(function, grad);
		}
		return function;
	}

	public void BFGSUpdate(Optimizable function, SimpleMatrix grad) {

		SimpleMatrix pt = B.mult(grad.scale(-1.));

		epoch++;

		// TODO: optimize this!
		SimpleMatrix scalemat = new SimpleMatrix(dimension, 1);
		for (int i = 0; i < dimension; i++) {
			scalemat.set(i, annealingSchedule.learningRate(epoch, i) / c);
		}
		SimpleMatrix st = pt.elementMult(scalemat);

		SimpleMatrix paramVec = new SimpleMatrix(new EJMLMatrixDecorator(
				function.getVector()));

		paramVec = paramVec.plus(st);
		LinearVector newParameters = LinearVectorFactory.getVector(dimension);

		for (int i = 0; i < dimension; i++) {
			newParameters.resetValue(i, paramVec.get(i, 0));
		}

		SimpleMatrix yt = new SimpleMatrix(new EJMLMatrixDecorator(
				function.computeGradient(newParameters))).minus(grad).plus(
				st.scale(lambda));

		double numerator = st.dot(yt);
		double denominator = yt.dot(yt);

		double[] diag = new double[dimension];
		Arrays.fill(diag, numerator / denominator);

		B = SimpleMatrix.diag(diag);

		double psi = 1. / st.dot(yt);
		SimpleMatrix ident = SimpleMatrix.identity(dimension);
		SimpleMatrix first = ident.minus(st.scale(psi).mult(yt.transpose()));
		SimpleMatrix second = ident.minus(yt.scale(psi).mult(st.transpose()));
		SimpleMatrix third = st.scale(c * psi).mult(st.transpose());

		B = first.mult(B).mult(second).plus(third);
		function.setParameters(newParameters);
	}

	@Override
	protected Optimizable updateModelWithRegularization(Optimizable function,
			LinearVector regularizationGradient) {
		SimpleMatrix grad = new SimpleMatrix(new EJMLMatrixDecorator(
				regularizationGradient));
		BFGSUpdate(function, grad.scale(-1));
		return function;
	}

}
