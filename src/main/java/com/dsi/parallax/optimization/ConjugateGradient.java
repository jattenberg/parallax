/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;
import com.dsi.parallax.optimization.linesearch.BackTrackLineSearch;
import com.dsi.parallax.optimization.linesearch.GradientLineOptimizer;

public class ConjugateGradient extends AbstractOptimizer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4073033470547408803L;
	private boolean converged = false;
	private GradientOptimizable optimizable;
	private GradientLineOptimizer lineMaximizer;

	// xxx If this is too big, we can get inconsistent value and gradient in
	// MaxEntTrainer
	// Investigate!!!
	double initialStepSize = 0.01;

	// "eps" is a small number to recitify the special case of converging
	// to exactly zero function value
	private GradientOptimizerEvaluator eval;

	public ConjugateGradient(GradientOptimizable function,
			double initialStepSize) {
		this.initialStepSize = initialStepSize;
		this.optimizable = function;
		this.lineMaximizer = new BackTrackLineSearch(function);
	}

	public ConjugateGradient(GradientOptimizable function) {
		this(function, 0.01);
	}

	public Optimizable getOptimizable() {
		return this.optimizable;
	}

	public boolean isConverged() {
		return converged;
	}

	public void setEvaluator(GradientOptimizerEvaluator eval) {
		this.eval = eval;
	}

	public void setLineMaximizer(GradientLineOptimizer lineMaximizer) {
		this.lineMaximizer = lineMaximizer;
	}

	public void setInitialStepSize(double initialStepSize) {
		this.initialStepSize = initialStepSize;
	}

	public double getInitialStepSize() {
		return this.initialStepSize;
	}

	public double getStepSize() {
		return step;
	}

	// The state of a conjugate gradient search
	double fp, gg, gam, dgg, step, fret;
	LinearVector xi, g, h;
	int j, iterations;

	@Override
	public boolean optimize(int numIterations) {
		if (converged)
			return true;

		double prevStepSize = initialStepSize;
		boolean searchingGradient = true;
		if (xi == null) {
			fp = optimizable.computeLoss();
			xi = optimizable.getValueGradient();
			g = LinearVectorFactory.getVector(xi);
			h = LinearVectorFactory.getVector(xi);
			step = initialStepSize;
			iterations = 0;
		}

		for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
			logger.info("ConjugateGradient: At iteration " + iterations
					+ ", cost = " + fp);
			try {
				prevStepSize = step;
				step = lineMaximizer.optimize(xi, step);
			} catch (IllegalArgumentException e) {
				logger.warn("ConjugateGradient caught " + e.toString());
				OptimizationUtils
						.testValueAndGradientCurrentParameters(optimizable);
				OptimizationUtils.testValueAndGradientInDirection(optimizable,
						xi);
				// return this.maximize (maxable, numIterations);
			}
			if (step == 0) {
				if (searchingGradient) {
					logger.info("ConjugateGradient converged: Line maximizer got step 0 in gradient direction.  "
							+ "Gradient absNorm=" + xi.L1Norm());
					converged = true;
					return true;
				} else
					logger.info("Line maximizer got step 0.  Probably pointing up hill.  Resetting to gradient.  "
							+ "Gradient absNorm=" + xi.L1Norm());
				// Copied from above (how to code this better? I want GoTo)
				fp = optimizable.computeLoss();
				xi = optimizable.getValueGradient();
				searchingGradient = true;
				g = LinearVectorFactory.getVector(xi);
				h = LinearVectorFactory.getVector(xi);
				step = prevStepSize;
				continue;
			}
			fret = optimizable.computeLoss();
			// This termination provided by "Numeric Recipes in C".
			if (checkValueTerminationCondition(fret, fp)) {
				logger.info("ConjugateGradient converged: old value= " + fp
						+ " new value= " + fret + " tolerance=" + tolerance);
				converged = true;
				return true;
			}
			fp = fret;
			xi = optimizable.getValueGradient();

			logger.info("Gradient infinityNorm = " + xi.LInfinityNorm());
			// This termination provided by McCallum
			if (xi.LInfinityNorm() < tolerance) {
				logger.info("ConjugateGradient converged: maximum gradient component "
						+ xi.LInfinityNorm() + ", less than " + tolerance);
				converged = true;
				return true;
			}

			dgg = gg = 0.0;
			double gj, xj;
			for (j = 0; j < xi.size(); j++) {
				gj = g.getValue(j);
				gg += gj * gj;
				xj = -xi.getValue(j);
				dgg = (xj + gj) * xj;
			}
			if (gg == 0.0) {
				logger.info("ConjugateGradient converged: gradient is exactly zero.");
				converged = true;
				return true; // In unlikely case that gradient is exactly zero,
								// then we are done
			}
			gam = dgg / gg;

			double hj;
			for (j = 0; j < xi.size(); j++) {
				xj = xi.getValue(j);
				g.resetValue(j, xj);
				hj = h.getValue(j);
				hj = xj + gam * hj;
				h.resetValue(j, hj);
			}

			if (VectorUtils.isNaN(h))
				throw new IllegalStateException("h should not be NaN!");
			xi = LinearVectorFactory.getVector(h);
			searchingGradient = false;

			iterations++;
			if (iterations > maxIterations) {
				logger.warn("Too many iterations in ConjugateGradient.java");
				converged = true;
				return true;
				// throw new IllegalStateException ("Too many iterations.");
			}

			if (eval != null)
				eval.evaluate(optimizable, iterations);
		}
		return false;
	}

	public void reset() {
		xi = null;
	}
}
