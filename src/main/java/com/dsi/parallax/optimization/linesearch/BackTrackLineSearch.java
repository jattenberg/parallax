/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.linesearch;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;
import com.dsi.parallax.optimization.GradientOptimizable;

//"Line Searches and Backtracking", p385, "Numeric Recipes in C"

public class BackTrackLineSearch extends AbstractLineOptimizer implements
		GradientLineOptimizer {
	private final int maxIterations = 100;
	private final double stpmax = 100;

	// termination conditions: either
	// a) abs(delta x/x) < REL_TOLX for all coordinates
	// b) abs(delta x) < ABS_TOLX for all coordinates
	// c) sufficient function increase (uses ALF)
	private double relTolx = 1e-7;
	private final double ALF = 1e-4;
	private double absTolx = 1e-4;

	private GradientOptimizable function;

	public BackTrackLineSearch(GradientOptimizable optimizable) {
		this.function = optimizable;
	}

	/**
	 * Sets the tolerance of relative diff in function value. Line search
	 * converges if <tt>abs(delta x / x) < tolx</tt> for all coordinates.
	 */
	public void setRelTolx(double tolx) {
		relTolx = tolx;
	}

	/**
	 * Sets the tolerance of absolute diff in function value. Line search
	 * converges if <tt>abs(delta x) < tolx</tt> for all coordinates.
	 */
	public void setAbsTolx(double tolx) {
		absTolx = tolx;
	}

	// initialStep is ignored. This is b/c if the initial step is not 1.0,
	// it sometimes confuses the backtracking for reasons I don't
	// understand. (That is, the jump gets LARGER on iteration 1.)

	// returns fraction of step size (alam) if found a good step
	// returns 0.0 if could not step in direction
	@Override
	public double optimize(LinearVector line, double initialStep) {

		LinearVector g, x, oldParameters;
		double slope, temp, test, alamin, alam, alam2, tmplam;
		double rhs1, rhs2, a, b, disc, oldAlam;
		double f, fold, f2;
		int size = function.getNumParameters();
		x = function.getVector();
		oldParameters = LinearVectorFactory.getVector(x);

		g = function.getValueGradient();
		alam2 = tmplam = 0.0;
		f2 = fold = function.computeLoss();

		if (logger.isInfoEnabled()) {
			logger.info("ENTERING BACKTRACK\n");
			logger.info("Entering BackTrackLnSrch, value=" + fold
					+ ",\ndirection.oneNorm:" + line.L1Norm()
					+ "  direction.infNorm:" + line.LInfinityNorm());
		}
		double sum = line.L2Norm();
		if (sum > stpmax) {
			logger.warn("attempted step too big. scaling: sum=" + sum
					+ ", stpmax=" + stpmax);
			line.timesEquals(stpmax / sum);
		}

		slope = VectorUtils.dotProduct(g, line);
		logger.info("slope=" + slope);

		if (slope <= 0)
			throw new RuntimeException(
					"invalid slope, must be positive. given: " + slope);

		// find maximum lambda
		// converge when (delta x) / x < REL_TOLX for all coordinates.
		// the largest step size that triggers this threshold is
		// precomputed and saved in alamin
		test = 0.0;
		for (int i = 0; i < size; i++) {
			temp = Math.abs(line.getValue(i))
					/ Math.max(Math.abs(oldParameters.getValue(i)), 1.0);
			if (temp > test)
				test = temp;
		}

		alamin = relTolx / test;
		alam = 1.0;
		oldAlam = 0.0;
		int iteration = 0;
		// look for step size in direction given by "line"
		for (iteration = 0; iteration < maxIterations; iteration++) {
			// x = oldParameters + alam*line
			// initially, alam = 1.0, i.e. take full Newton step
			logger.info("BackTrack loop iteration " + iteration + ": alam="
					+ alam + " oldAlam=" + oldAlam);
			logger.info("before step, x.1norm: " + x.L1Norm() + "\nalam: "
					+ alam + "\noldAlam: " + oldAlam);
			x.plusEqualsVectorTimes(line, alam - oldAlam); // step
			logger.info("after step, x.1norm: " + x.L1Norm());

			// check for convergence
			// convergence on delta x
			if ((alam < alamin)
					|| VectorUtils.smallAbsDiff(oldParameters, x, absTolx)) {
				function.setParameters(oldParameters);
				f = function.computeLoss();
				logger.warn("EXITING BACKTRACK: Jump too small (alamin="
						+ alamin + "). Exiting and using xold. Value=" + f);
				return 0.0;
			}

			function.setParameters(x);
			oldAlam = alam;
			f = function.computeLoss();

			logger.info("value=" + f);

			// sufficient function increase (Wolf condition)
			if (f >= fold + ALF * alam * slope) {

				logger.info("EXITING BACKTRACK: value=" + f);

				if (f < fold)
					throw new IllegalStateException(
							"Function did not increase: f=" + f + " < " + fold
									+ "=fold");
				return alam;
			}
			// if value is infinite, i.e. we've
			// jumped to unstable territory, then scale down jump
			else if (Double.isInfinite(f) || Double.isInfinite(f2)) {
				logger.warn("Value is infinite after jump " + oldAlam + ". f="
						+ f + ", f2=" + f2 + ". Scaling back step size...");
				tmplam = .2 * alam;
				if (alam < alamin) { // convergence on delta x
					function.setParameters(oldParameters);
					f = function.computeLoss();
					logger.warn("EXITING BACKTRACK: Jump too small. Exiting and using xold. Value="
							+ f);
					return 0.0;
				}
			} else { // backtrack
				if (alam == 1.0) // first time through
					tmplam = -slope / (2.0 * (f - fold - slope));
				else {
					rhs1 = f - fold - alam * slope;
					rhs2 = f2 - fold - alam2 * slope;

					a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2))
							/ (alam - alam2);
					b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2
							/ (alam2 * alam2))
							/ (alam - alam2);
					if (a == 0.0)
						tmplam = -slope / (2.0 * b);
					else {
						disc = b * b - 3.0 * a * slope;
						if (disc < 0.0) {
							tmplam = .5 * alam;
						} else if (b <= 0.0)
							tmplam = (-b + Math.sqrt(disc)) / (3.0 * a);
						else
							tmplam = -slope / (b + Math.sqrt(disc));
					}
					if (tmplam > .5 * alam)
						tmplam = .5 * alam; // lambda <= .5 lambda_1
				}
			}
			alam2 = alam;
			f2 = f;
			logger.info("tmplam:" + tmplam);
			alam = Math.max(tmplam, .1 * alam); // lambda >= .1*Lambda_1
		}
		if (iteration >= maxIterations)
			throw new IllegalStateException("Too many iterations.");
		return 0.0;
	}
}
