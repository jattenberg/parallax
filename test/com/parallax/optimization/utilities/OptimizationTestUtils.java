/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.utilities;

import static com.google.common.base.Preconditions.checkArgument;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import com.parallax.ml.util.MLUtils;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.optimization.Gradient;
import com.parallax.optimization.Optimizable;
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;

public class OptimizationTestUtils {

	public static double[] minimum = new double[] { 5, -3 };
	public static double[] minimum2 = new double[] { 5. / 6. };

	public static double[] functionGradient(double[] X) {
		// d/dx x1^2 - 4x1 + 2*x1x2 + 2x2^22 + 2*x2 + 14
		checkArgument(
				X.length == 2,
				"input vector must be of length 2, input: "
						+ Arrays.toString(X));

		double[] out = new double[2];
		out[0] = 2 * X[0] - 4 + 2 * X[1];
		out[1] = 2 * X[0] + 4 * X[1] + 2;

		return out;
	}

	public static double distToMinimum(double[] X) {
		checkArgument(
				X.length == 2,
				"input vector must be of length 2, input: "
						+ Arrays.toString(X));

		return Math.sqrt(Math.pow(X[0] - minimum[0], 2)
				+ Math.pow(X[1] - minimum[1], 2));
	}

	public static double distToMinimum2(double[] X) {
		checkArgument(
				X.length == 1,
				"input vector must be of length 1, input: "
						+ Arrays.toString(X));

		return Math.sqrt(Math.pow(X[0] - minimum2[0], 2));
	}

	@Test
	public void testMinimum() {
		FunctionOne f1 = new FunctionOne();
		FunctionTwo f2 = new FunctionTwo();
		assertEquals(distToMinimum(minimum), 0, 0.000001);
		assertTrue(f1
				.functionValue(LinearVectorFactory.getDenseVector(minimum)) < f1
				.functionValue(LinearVectorFactory.getDenseVector(new double[] {
						1, 1 })));
		assertEquals(distToMinimum2(minimum2), 0, 0.000001);
		assertTrue(f2.functionValue2(LinearVectorFactory
				.getDenseVector(minimum2)) < f2
				.functionValue2(LinearVectorFactory
						.getDenseVector(new double[] { 5 })));

	}

	@Test
	public void testGradient() {
		double[] grad = functionGradient(new double[] { 4, -4 });
		assertEquals(-4, grad[0], 0.00001);
		assertEquals(-6, grad[1], 0.00001);
	}

	public static class FunctionOne implements Optimizable {

		private LinearVector parameters;

		public FunctionOne() {
			parameters = LinearVectorFactory.getDenseVector(2);
		}

		@Override
		public int getNumParameters() {
			return 2;
		}

		private LinearVector functionGradient(LinearVector params) {
			// d/dx x1^2 - 4x1 + 2*x1x2 + 2x2^22 + 2*x2 + 14
			checkArgument(params.size() == 2,
					"input vector must be of length 2, input: " + params);

			LinearVector out = LinearVectorFactory.getDenseVector(2);
			out.resetValue(0,
					2 * params.getValue(0) - 4 + 2 * params.getValue(1));
			out.resetValue(1, 2 * params.getValue(0) + 4 * params.getValue(1)
					+ 2);

			return out;
		}

		private double functionValue(LinearVector params) {
			// x1^2 - 4x1 + 2*x1x2 + 2x2^22 + 2*x2 + 14
			checkArgument(params.size() == 2,
					"input vector must be of length 2, input: " + params);
			return params.getValue(0) * params.getValue(0) - 4
					* params.getValue(0) + 2 * params.getValue(0)
					* params.getValue(1) + 2 * params.getValue(1)
					* params.getValue(1) + 2 * params.getValue(1) + 14;
		}

		@Override
		public LinearVector getVector() {
			return parameters;
		}

		@Override
		public double getParameter(int index) {
			return parameters.getValue(index);
		}

		@Override
		public void setParameter(int index, double value) {
			parameters.resetValue(index, value);
		}

		@Override
		public void setParameters(LinearVector params) {
			parameters = params;
		}

		@Override
		public Gradient computeGradient() {
			return computeGradient(parameters);
		}

		@Override
		public Gradient computeGradient(LinearVector params) {
			return new Gradient(functionGradient(params), computeLoss(params));
		}

		@Override
		public double computeLoss() {
			return computeLoss(parameters);
		}

		public double computeLoss(LinearVector params) {
			return Math.sqrt(Math.pow(params.getValue(0) - minimum[0], 2)
					+ Math.pow(params.getValue(1) - minimum[1], 2));
		}
	}

	public static class FunctionTwo implements Optimizable {

		private LinearVector parameters;

		public FunctionTwo() {
			parameters = LinearVectorFactory.getDenseVector(1);
		}

		public double functionValue2(LinearVector params) {
			// x1^2 - 4x1 + 2*x1x2 + 2x2^22 + 2*x2 + 14
			checkArgument(params.size() == 1,
					"input vector must be of length 1, input: " + params);

			return 3 * params.getValue(0) * params.getValue(0) - 5
					* params.getValue(0) + 2;
		}

		public LinearVector functionGradient2(LinearVector params) {
			// d/dx x1^2 - 4x1 + 2*x1x2 + 2x2^22 + 2*x2 + 14
			checkArgument(params.size() == 1,
					"input vector must be of length 1, input: " + params);

			LinearVector out = LinearVectorFactory.getDenseVector(1);
			out.resetValue(0, 6 * params.getValue(0) - 5);

			return out;
		}

		@Override
		public int getNumParameters() {
			return 1;
		}

		@Override
		public LinearVector getVector() {
			return parameters;
		}

		@Override
		public double getParameter(int index) {
			return parameters.getValue(index);
		}

		@Override
		public void setParameter(int index, double value) {
			parameters.resetValue(index, value);
		}

		@Override
		public void setParameters(LinearVector params) {
			parameters = params;
		}

		@Override
		public Gradient computeGradient() {
			return computeGradient(parameters);
		}

		@Override
		public Gradient computeGradient(LinearVector params) {
			return new Gradient(functionGradient2(params), computeLoss(params));
		}

		@Override
		public double computeLoss() {
			return computeLoss(parameters);
		}

		public double computeLoss(LinearVector params) {
			return Math.sqrt(Math.pow(params.getValue(0) - minimum2[0], 2));
		}
	}

	public static class RandomFunction implements Optimizable {

		private LinearVector parameters;

		public RandomFunction(int dims) {
			parameters = LinearVectorFactory.getVector(dims);
			for (int i = 0; i < dims; i++)
				parameters.resetValue(i, MLUtils.GENERATOR.nextDouble());
		}

		@Override
		public int getNumParameters() {
			return parameters.size();
		}

		@Override
		public LinearVector getVector() {
			return parameters;
		}

		@Override
		public double getParameter(int index) {
			return parameters.getValue(index);
		}

		@Override
		public void setParameter(int index, double value) {
			parameters.resetValue(index, value);

		}

		@Override
		public void setParameters(LinearVector params) {
			this.parameters = params;
		}

		@Override
		public Gradient computeGradient() {
			return computeGradient(parameters);
		}

		@Override
		public Gradient computeGradient(LinearVector params) {
			return new Gradient(
					LinearVectorFactory.getDenseVector(getNumParameters()),
					computeLoss(params));
		}

		@Override
		public double computeLoss() {
			return 0;
		}

		@Override
		public double computeLoss(LinearVector params) {
			return 0;
		}
	}

	public static class ConstantFunction implements Optimizable {

		private LinearVector parameters;

		public ConstantFunction(int dims, double value) {
			parameters = LinearVectorFactory.getVector(dims);
			for (int i = 0; i < dims; i++)
				parameters.resetValue(i, value);
		}

		@Override
		public int getNumParameters() {
			return parameters.size();
		}

		@Override
		public LinearVector getVector() {
			return parameters;
		}

		@Override
		public double getParameter(int index) {
			return parameters.getValue(index);
		}

		@Override
		public void setParameter(int index, double value) {
			parameters.resetValue(index, value);

		}

		@Override
		public void setParameters(LinearVector params) {
			this.parameters = params;
		}

		@Override
		public Gradient computeGradient() {
			return computeGradient(parameters);
		}

		@Override
		public Gradient computeGradient(LinearVector params) {
			return new Gradient(
					LinearVectorFactory.getDenseVector(getNumParameters()),
					computeLoss(params));
		}

		@Override
		public double computeLoss() {
			return 3;
		}

		@Override
		public double computeLoss(LinearVector params) {
			return 3;
		}
	}

	public static AnnealingScheduleConfigurableBuilder buildConstantAnnealingScheduleBuilder() {
		return buildConstantAnnealingScheduleBuilder(0.1);
	}

	public static AnnealingScheduleConfigurableBuilder buildConstantAnnealingScheduleBuilder(
			double rate) {
		return AnnealingScheduleConfigurableBuilder
				.configureForConstantRate(rate);
	}

	public static AnnealingScheduleConfigurableBuilder buildInverseAnnealingScheduleBuilder() {
		return buildInverseAnnealingScheduleBuilder(15, 0.999999);
	}

	public static AnnealingScheduleConfigurableBuilder buildInverseAnnealingScheduleBuilder(
			double rate, double decay) {
		return AnnealingScheduleConfigurableBuilder.configureForInverseDecay(
				rate, decay);
	}

	public static AnnealingScheduleConfigurableBuilder buildExponentialAnnealingScheduleBuilder() {
		return buildExponentialAnnealingScheduleBuilder(.1, 0.999999);
	}

	public static AnnealingScheduleConfigurableBuilder buildExponentialAnnealingScheduleBuilder(
			double rate, double base) {
		return AnnealingScheduleConfigurableBuilder
				.configureForExponentialDecay(rate, base);
	}
}
