/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.util.MLUtils;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.regularization.TruncationType;
import com.parallax.optimization.utilities.OptimizationTestUtils;
import com.parallax.optimization.utilities.OptimizationTestUtils.ConstantFunction;
import com.parallax.optimization.utilities.OptimizationTestUtils.FunctionOne;
import com.parallax.optimization.utilities.OptimizationTestUtils.FunctionTwo;
import com.parallax.optimization.utilities.OptimizationTestUtils.RandomFunction;

public class TestStochasaticGradientDescent {

	@Test
	public void testGaussianRegularization() {
		SGDBuilder sgdBuilder = new SGDBuilder(10, false);

		sgdBuilder.setGaussianWeight(3);
		StochasaticGradientDescent sgd = sgdBuilder.build();

		assertEquals(sgd.getRegularizationWeight(), 1., 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.GAUSSIAN), 3, 0);

		ConstantFunction funct = new ConstantFunction(10, 5.);
		double l2norm = funct.getVector().L2Norm();

		for (int i = 0; i < 50; i++) {
			sgd.update(funct);
			double newl2norm = funct.getVector().L2Norm();
			assertTrue(newl2norm < l2norm);
			l2norm = newl2norm;
		}

	}

	@Test
	public void testLaplaceRegularization() {
		SGDBuilder sgdBuilder = new SGDBuilder(10, false);

		sgdBuilder.setLaplaceWeight(3);
		StochasaticGradientDescent sgd = sgdBuilder.build();

		assertEquals(sgd.getRegularizationWeight(), 1., 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.LAPLACE), 3, 0);

		ConstantFunction funct = new ConstantFunction(10, 5.);
		double l2norm = funct.getVector().L2Norm();

		for (int i = 0; i < 50; i++) {
			sgd.update(funct);
			double newl2norm = funct.getVector().L2Norm();
			assertTrue(newl2norm < l2norm);
			l2norm = newl2norm;
		}

	}

	@Test
	public void testCauchyRegularization() {
		SGDBuilder sgdBuilder = new SGDBuilder(10, false);

		sgdBuilder.setCauchyWeight(3);
		StochasaticGradientDescent sgd = sgdBuilder.build();

		assertEquals(sgd.getRegularizationWeight(), 1., 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.CAUCHY), 3, 0);

		ConstantFunction funct = new ConstantFunction(10, 5.);
		double l2norm = funct.getVector().L2Norm();

		for (int i = 0; i < 50; i++) {
			sgd.update(funct);
			double newl2norm = funct.getVector().L2Norm();
			assertTrue(newl2norm < l2norm);
			l2norm = newl2norm;
		}

	}

	@Test
	public void testSquaredRegularization() {
		SGDBuilder sgdBuilder = new SGDBuilder(10, false);

		sgdBuilder.setSquaredWeight(3);
		StochasaticGradientDescent sgd = sgdBuilder.build();

		assertEquals(sgd.getRegularizationWeight(), 1., 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.SQUARED), 3, 0);

		ConstantFunction funct = new ConstantFunction(10, 5.);
		double l2norm = funct.getVector().L2Norm();

		for (int i = 0; i < 50; i++) {
			sgd.update(funct);
			double newl2norm = funct.getVector().L2Norm();
			assertTrue(newl2norm < l2norm);
			l2norm = newl2norm;
		}

	}

	@Test
	public void testMultiRegularization() {
		SGDBuilder sgdBuilder = new SGDBuilder(10, false);

		sgdBuilder.setGaussianWeight(3).setCauchyWeight(1.2)
				.setLaplaceWeight(.55).setRegularizationWeight(0.1);
		StochasaticGradientDescent sgd = sgdBuilder.build();

		assertEquals(sgd.getRegularizationWeight(), 0.1, 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.GAUSSIAN), 3, 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.CAUCHY), 1.2, 0);
		assertEquals(
				sgd.getCoefficientWeights().get(
						LinearCoefficientLossType.LAPLACE), .55, 0);

		ConstantFunction funct = new ConstantFunction(10, 5.);
		double l2norm = funct.getVector().L2Norm();

		for (int i = 0; i < 50; i++) {
			sgd.update(funct);
			double newl2norm = funct.getVector().L2Norm();
			assertTrue(newl2norm < l2norm);
			l2norm = newl2norm;
		}

	}

	@Test
	public void testGradientTruncation() {
		for (TruncationType type : TruncationType.values()) {
			SGDBuilder sgdBuilder = new SGDBuilder(1000, false);
			TruncationConfigurableBuilder builder = new TruncationConfigurableBuilder()
					.setAlpha(type == TruncationType.MODDUCHI ? 1050 : 500)
					.setThreshold(.01).setPeriod(1).setTruncationType(type);

			sgdBuilder.setGradientTruncationBuilder(builder);
			RandomFunction funct = new RandomFunction(1000);

			StochasaticGradientDescent sgd = sgdBuilder.build();

			double beforeTwoNorm = funct.getVector().L2Norm();
			double beforeOneNorm = funct.getVector().L1Norm();

			sgd.update(funct);
			double afterTwoNorm = funct.getVector().L2Norm();
			double afterOneNorm = funct.getVector().L1Norm();

			if (type.equals(TruncationType.NONE)) {
				assertEquals(beforeOneNorm, afterOneNorm, .00000001);
				assertEquals(beforeTwoNorm, afterTwoNorm, .00000001);
			} else {
				assertTrue(afterOneNorm < beforeOneNorm);
				assertTrue(afterTwoNorm < beforeTwoNorm);
			}

		}
	}

	@Test
	public void TestOptimizes() {
		SGDBuilder builder = new SGDBuilder(2, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildConstantAnnealingScheduleBuilder(0.1));
		StochasaticGradientDescent sgd = builder.build();

		FunctionOne f1 = new FunctionOne();
		double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
				.getW());

		for (int i = 0; i < 150; i++) {

			double loss = distToMin;
			sgd.update(f1);
			distToMin = f1.computeLoss();
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}

		assertEquals(0, distToMin, 0.001);
	}

	@Test
	public void TestOptimizes2() {
		SGDBuilder builder = new SGDBuilder(1, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildConstantAnnealingScheduleBuilder(0.1));
		StochasaticGradientDescent sgd = builder.build();

		FunctionTwo f2 = new FunctionTwo();
		double distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
				.getW());
		for (int i = 0; i < 25; i++) {
			double loss = distToMin;
			sgd.update(f2);
			distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
					.getW());
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}
		assertEquals(f2.getParameter(0), OptimizationTestUtils.minimum2[0],
				0.0000001);
	}

	// different annealing schedules
	public void TestInverseAnnealOptimize() {
		SGDBuilder builder = new SGDBuilder(2, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildInverseAnnealingScheduleBuilder(0.5, 0.2));
		StochasaticGradientDescent sgd = builder.build();

		FunctionOne f1 = new FunctionOne();
		double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
				.getW());

		for (int i = 0; i < 150; i++) {

			double loss = distToMin;
			sgd.update(f1);
			distToMin = f1.computeLoss();
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}

		assertEquals(0, distToMin, 0.001);
	}

	@Test
	public void TestInverseAnnealOptimize2() {
		SGDBuilder builder = new SGDBuilder(1, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildInverseAnnealingScheduleBuilder(0.1, 50));
		StochasaticGradientDescent sgd = builder.build();
		FunctionTwo f2 = new FunctionTwo();
		double distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
				.getW());
		for (int i = 0; i < 45; i++) {
			double loss = distToMin;
			sgd.update(f2);
			distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
					.getW());
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}
		assertEquals(f2.getParameter(0), OptimizationTestUtils.minimum2[0],
				0.0000001);
	}

	@Test
	public void TestExponentialAnnealOptimize() {
		SGDBuilder builder = new SGDBuilder(2, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildExponentialAnnealingScheduleBuilder());
		StochasaticGradientDescent sgd = builder.build();
		FunctionOne f1 = new FunctionOne();
		double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
				.getW());

		for (int i = 0; i < 150; i++) {

			double loss = distToMin;
			sgd.update(f1);
			distToMin = f1.computeLoss();
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}

		assertEquals(0, distToMin, 0.001);
	}

	@Test
	public void TestExponentialAnnealOptimize2() {
		SGDBuilder builder = new SGDBuilder(1, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildExponentialAnnealingScheduleBuilder());
		StochasaticGradientDescent sgd = builder.build();
		FunctionTwo f2 = new FunctionTwo();
		double distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
				.getW());
		for (int i = 0; i < 25; i++) {
			double loss = distToMin;
			sgd.update(f2);
			distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
					.getW());
			assertTrue(distToMin < loss
					|| MLUtils.floatingPointEquals(distToMin, 0));
		}
		assertEquals(f2.getParameter(0), OptimizationTestUtils.minimum2[0],
				0.0000001);
	}
}
