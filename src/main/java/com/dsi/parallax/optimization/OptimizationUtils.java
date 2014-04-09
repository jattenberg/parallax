/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import org.apache.log4j.Logger;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;

public class OptimizationUtils {
	private static Logger logger = Logger.getLogger(OptimizationUtils.class);

	private OptimizationUtils() {
	}

	public static double testValueAndGradientCurrentParameters(
			GradientOptimizable maxable) {
		return testValueAndGradientCurrentParameters(maxable, -1);
	}

	public static double testValueAndGradientCurrentParameters(
			GradientOptimizable maxable, int numComponents) {
		LinearVector parameters = LinearVectorFactory.getVector(maxable
				.getVector());
		double value = maxable.computeLoss();
		// the gradient from the maximizable function
		LinearVector analyticGradient = maxable.getValueGradient();
		// the gradient calculate from the slope of the value
		LinearVector empiricalGradient = maxable.getValueGradient();

		// This setting of epsilon should make the individual elements of
		// the analytical gradient and the empirical gradient equal. This
		// simplifies the comparison of the individual dimensions of the
		// gradient and thus makes debugging easier.
		// cas: However, avoid huge epsilon if norm of analytic gradient is
		// close to 0.
		// Next line used to be: double norm = Math.max (0.1,
		// MatrixOps.twoNorm(analyticGradient));
		// but if all the components of the analyticalGradient are very small,
		// the squaring in the
		// twoNorm causes epsilon to be too large. -AKM
		double norm = Math.max(0.1, analyticGradient.L1Norm());
		double epsilon = 0.1 / norm;
		double tolerance = epsilon * 5;
		logger.info("epsilon = " + epsilon + " tolerance=" + tolerance);

		int sampleParameterInterval = -1;
		if (numComponents > 0) {
			sampleParameterInterval = Math.max(1, parameters.size()
					/ numComponents);
			logger.info("Will check every " + sampleParameterInterval
					+ "-th component.");
		}

		// Check each direction, perturb it, measure new value, and make
		// sure it agrees with the gradient from
		// maxable.getValueGradient()
		for (int i = 0; i < parameters.size(); i++) {
			if ((parameters.size() >= sampleParameterInterval)
					&& (i % sampleParameterInterval != 0))
				continue;

			double param = parameters.getValue(i);
			parameters.resetValue(i, param + epsilon);

			maxable.setParameters(parameters);
			double epsValue = maxable.computeLoss();
			double slope = (epsValue - value) / epsilon;
			logger.info("value=" + value + " epsValue=" + epsValue + " slope["
					+ i + "] = " + slope + " gradient[]="
					+ analyticGradient.getValue(i));
			if (Double.isNaN(slope))
				throw new IllegalStateException("slope is NaN");
			logger.info("TestMaximizable checking singleIndex " + i
					+ ": gradient slope = " + analyticGradient.getValue(i)
					+ ", value+epsilon slope = " + slope
					+ ": slope difference = "
					+ (slope - analyticGradient.getValue(i)));
			// No negative below because the gradient points in the direction
			// of maximizing the function.
			empiricalGradient.resetValue(i, slope);
			parameters.resetValue(i, param);
		}
		// Normalize the matrices to have the same L2 length
		logger.info("analyticGradient.twoNorm = " + analyticGradient.L2Norm());
		logger.info("empiricalGradient.twoNorm = " + empiricalGradient.L2Norm());
		analyticGradient.timesEquals(1. / analyticGradient.L2Norm());
		analyticGradient.timesEquals(1. / empiricalGradient.L2Norm());

		// Return the angle between the two vectors, in radians
		double dot = VectorUtils
				.dotProduct(analyticGradient, empiricalGradient);
		if (MLUtils.floatingPointEquals(dot, 1.)) {
			logger.info("TestMaximizable angle is zero.");
			return 0.0;
		} else {
			double angle = Math.acos(dot);

			logger.info("TestMaximizable angle = " + angle);
			if (Math.abs(angle) > tolerance)
				throw new IllegalStateException(
						"Gradient/Value mismatch: angle=" + angle + " tol: "
								+ tolerance);
			if (Double.isNaN(angle))
				throw new IllegalStateException(
						"Gradient/Value error: angle is NaN!");

			return angle;
		}
	}

	public static double testValueAndGradientInDirection(
			GradientOptimizable maxable, LinearVector direction) {
		int numParameters = maxable.getNumParameters();
		if (direction.size() != numParameters)
			throw new IllegalStateException("numParams (" + numParameters
					+ ") and direction gradient size (" + direction.size()
					+ ") must match");

		LinearVector parameters = LinearVectorFactory.getVector(maxable
				.getVector());
		LinearVector oldParameters = LinearVectorFactory.getVector(parameters);
		LinearVector normalizedDirection = LinearVectorFactory
				.getVector(direction);

		normalizedDirection.absNormalize();

		double value = maxable.computeLoss();
		// the gradient from the optimizable function

		LinearVector analyticGradient = maxable.getValueGradient();
		// the gradient calculate from the slope of the value
		// This setting of epsilon should make the individual elements of
		// the analytical gradient and the empirical gradient equal. This
		// simplifies the comparison of the individual dimensions of the
		// gradient and thus makes debugging easier.
		double directionGradient = VectorUtils.dotProduct(analyticGradient,
				normalizedDirection);
		double epsilon = 0.1 / analyticGradient.L1Norm();
		double tolerance = 0.00001 * directionGradient; // this was
														// "epsilon * 5";
		logger.info("epsilon = " + epsilon + " tolerance=" + tolerance);
		parameters.plusEqualsVectorTimes(normalizedDirection, epsilon);

		maxable.setParameters(parameters);
		double epsValue = maxable.computeLoss();
		double slope = (epsValue - value) / epsilon;
		logger.info("value=" + value + " epsilon=" + epsilon + " epsValue="
				+ epsValue + " slope = " + slope + " gradient="
				+ directionGradient);
		if (Double.isNaN(slope))
			throw new IllegalStateException("slope is NaN");

		double slopeDifference = Math.abs(slope - directionGradient);
		logger.info("TestMaximizable " + ": slope tolerance = " + tolerance
				+ ": gradient slope = " + directionGradient
				+ ", value+epsilon slope = " + slope + ": slope difference = "
				+ slopeDifference);
		maxable.setParameters(oldParameters);
		if (Math.abs(slopeDifference) > tolerance)
			throw new IllegalStateException("Slope difference "
					+ slopeDifference + " is greater than tolerance "
					+ tolerance);
		return slopeDifference;
	}
}
