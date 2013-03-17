package com.parallax.optimization.stochastic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.regularization.TruncationType;
import com.parallax.optimization.utilities.OptimizationTestUtils;
import com.parallax.optimization.utilities.OptimizationTestUtils.ConstantFunction;
import com.parallax.optimization.utilities.OptimizationTestUtils.FunctionOne;
import com.parallax.optimization.utilities.OptimizationTestUtils.FunctionTwo;
import com.parallax.optimization.utilities.OptimizationTestUtils.RandomFunction;

public class TestStochasticLBFGS {

	@Test
	public void TestOptimizes() {
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(2, false);

		builder.setLambda(0.1).setAnnealingScheduleConfigurableBuilder(
				OptimizationTestUtils.buildConstantAnnealingScheduleBuilder(1));

		StochasticLBFGS sgd = builder.build();
		FunctionOne f1 = new FunctionOne();
		double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
				.getW());

		for (int i = 0; i < 15; i++) {
			sgd.update(f1);
			distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
					.getW());
			// doesnt seem to converge uniformly, but still converges
		}

		assertEquals(0, distToMin, 0.001);
	}

	@Test
	public void TestOptimizes2() {
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(1, false);

		builder.setLambda(1).setAnnealingScheduleConfigurableBuilder(
				OptimizationTestUtils.buildConstantAnnealingScheduleBuilder());
		StochasticLBFGS sgd = builder.build();

		FunctionTwo f1 = new FunctionTwo();
		double distToMin = OptimizationTestUtils.distToMinimum2(f1.getVector()
				.getW());

		for (int i = 0; i < 1500; i++) {
			sgd.update(f1);
			distToMin = OptimizationTestUtils.distToMinimum2(f1.getVector()
					.getW());
			// doesnt seem to converge uniformly, but still converges
		}

		assertEquals(0, distToMin, 0.001);
	}

	// different annealing schedules
	@Test
	public void TestInverseAnnealOptimize() {
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(2, false);
		builder.setAnnealingScheduleConfigurableBuilder(OptimizationTestUtils
				.buildInverseAnnealingScheduleBuilder());
		StochasticLBFGS sgd = builder.build();

		FunctionOne f1 = new FunctionOne();
		double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
				.getW());

		for (int i = 0; i < 150; i++) {
			sgd.update(f1);
			distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
					.getW());
			// doesnt seem to converge uniformly, but still converges
		}

		assertEquals(0, distToMin, 0.001);
	}

	@Test
	public void testGaussianRegularization() {
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(10, false);

		builder.setGaussianWeight(3)
				.setRegularizationWeight(1.)
				.setAnnealingScheduleConfigurableBuilder(
						OptimizationTestUtils
								.buildConstantAnnealingScheduleBuilder(1));
		StochasticLBFGS sgd = builder.build();

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
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(10, false);

		builder.setLaplaceWeight(3).setAnnealingScheduleConfigurableBuilder(
				OptimizationTestUtils.buildConstantAnnealingScheduleBuilder(1));
		StochasticLBFGS sgd = builder.build();

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
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(10, false);

		builder.setCauchyWeight(3).setAnnealingScheduleConfigurableBuilder(
				OptimizationTestUtils.buildConstantAnnealingScheduleBuilder(1));
		StochasticLBFGS sgd = builder.build();

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
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(10, false);

		builder.setSquaredWeight(3).setAnnealingScheduleConfigurableBuilder(
				OptimizationTestUtils.buildConstantAnnealingScheduleBuilder(1));
		StochasticLBFGS sgd = builder.build();

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
		StochasticLBFGSBuilder builder = new StochasticLBFGSBuilder(10, false);

		builder.setGaussianWeight(3)
				.setCauchyWeight(1.2)
				.setLaplaceWeight(.55)
				.setRegularizationWeight(0.1)
				.setAnnealingScheduleConfigurableBuilder(
						OptimizationTestUtils
								.buildConstantAnnealingScheduleBuilder(1));
		StochasticLBFGS sgd = builder.build();

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
			StochasticLBFGSBuilder sgdBuilder = new StochasticLBFGSBuilder(50,
					false);
			TruncationConfigurableBuilder builder = new TruncationConfigurableBuilder()
					.setAlpha(type == TruncationType.MODDUCHI ? 1050 : 600)
					.setThreshold(.01).setPeriod(1).setTruncationType(type);

			sgdBuilder.setGradientTruncationBuilder(builder);
			RandomFunction funct = new RandomFunction(1000);

			StochasticLBFGS sgd = sgdBuilder.build();
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
}
