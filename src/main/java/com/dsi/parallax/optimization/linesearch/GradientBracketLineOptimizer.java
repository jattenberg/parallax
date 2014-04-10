/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.linesearch;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.VectorUtils;
import com.dsi.parallax.optimization.GradientOptimizable;

// Brents method using derivative information
// p405, "Numeric Recipes in C"

public class GradientBracketLineOptimizer extends AbstractLineOptimizer
		implements GradientLineOptimizer {

	int maxIterations = 50;
	GradientOptimizable optimizable;

	public GradientBracketLineOptimizer(GradientOptimizable function) {
		this.optimizable = function;
	}

	// TODO
	// This seems to work but is slower than BackTrackLineSearch. Why?

	// Return the last step size used.
	// "line" should point in the direction we want to move the parameters to
	// get
	// higher value.
	@Override
	public double optimize(LinearVector line, double initialStep) {

		assert (initialStep > 0);
		LinearVector parameters = LinearVectorFactory.getVector(optimizable
				.getVector());
		LinearVector gradient = optimizable.getValueGradient();

		// a=left, b=center, c=right, t=test
		double ax, bx, cx, tx; // steps (domain), these are deltas from initial
								// params!
		double ay, by, cy, ty; // costs (range)
		double ag, bg, tg; // projected gradients
		double ox; // the x step of the last function call

		tx = ax = bx = cx = ox = 0;
		ty = ay = by = cy = optimizable.computeLoss();

		tg = ag = bg = VectorUtils.dotProduct(gradient, line);
		// Make sure search-line points upward
		// logger.info ("Initial gradient = "+tg);
		if (ag <= 0) {
			throw new RuntimeException(
					"The search direction \"line\" does not point down uphill.  "
							+ "gradient.dotProduct(line)=" + ag
							+ ", but should be positive");
		}

		// Find an cx value where the gradient points the other way. Then
		// we will know that the (local) zero-gradient minimum falls
		// in between ax and cx.
		int iterations = 0;
		do {
			if (iterations++ > maxIterations)
				throw new IllegalStateException(
						"Exceeded maximum number allowed iterations searching for gradient cross-over.");
			// If we are still looking to cross the minimum, move ax towards it
			ax = bx;
			ay = by;
			ag = bg;
			// Save this (possibly) middle point; it might make an acceptable bx
			bx = tx;
			by = ty;
			bg = tg;
			if (tx == 0) {
				if (initialStep < 1.0) {
					tx = initialStep;
				} else {
					tx = 1.0;
				}
				// Sometimes the "suggested" initialStep is
				// very large and causes values to go to
				// infinity.
			} else {
				tx *= 3.0;
			}
			parameters.plusEqualsVectorTimes(line, tx - ox);
			optimizable.setParameters(parameters);
			ty = optimizable.computeLoss();
			gradient = optimizable.getValueGradient();
			tg = VectorUtils.dotProduct(gradient, line);

			ox = tx;
		} while (tg > 0);

		cx = tx;
		cy = ty;

		// We need to find a "by" that is less than both "ay" and "cy"
		if (Double.isNaN(by))
			throw new RuntimeException("by shouldnt be null");
		while (by <= ay || by <= cy || bx == ax) {
			// Last condition would happen if we did first while-loop only once
			if (iterations++ > maxIterations)
				throw new IllegalStateException(
						"Exceeded maximum number allowed iterations searching for bracketed minimum, iteratation count = "
								+ iterations);
			// xxx What should this tolerance be?
			// xxx I'm nervous that this is masking some assert()s below that
			// were previously failing.
			// If they were failing due to round-off error, that's OK, but if
			// not...
			if ((Math.abs(bg) < 100 || Math.abs(ay - by) < 10 || Math.abs(by
					- cy) < 10)
					&& bx != ax)
				// Magically, we are done
				break;

			// Instead make a version that finds the interpolating point by
			// fitting a parabola, and then jumps to that minimum. If the
			// actual y value is within "tolerance" of the parabola fit's
			// guess, then we are done, otherwise, use the parabola's x to
			// split the region, and try again.

			// There might be some cases where this will perform worse than
			// simply bisecting, as we do now, when the function is not at
			// all parabola shaped.

			// If the gradients ag and bg point in the same direction, then
			// the value by must be less than ay. And vice-versa for bg and cg.
			// assert (ax==bx || ((ag*bg)>=0 && by>ay) || (((bg*cg)>=0 &&
			// by>cy)));
			if (Double.isNaN(bg))
				throw new RuntimeException("bg shouldnt be null");

			if (bg > 0) {
				// the minimum is at higher x values than bx; drop ax
				assert (by >= ay);
				ax = bx;
				ay = by;
				ag = bg;
			} else {
				// the minimum is at lower x values than bx; drop cx
				assert (by >= cy);
				cx = bx;
				cy = by;
			}

			// Find a new mid-point
			bx = (ax + cx) / 2;
			parameters.plusEqualsVectorTimes(line, bx - ox);
			optimizable.setParameters(parameters);
			by = optimizable.computeLoss();
			if (Double.isNaN(by))
				throw new RuntimeException("by shouldnt be null");

			gradient = optimizable.getValueGradient();
			bg = VectorUtils.dotProduct(gradient, line);
			ox = bx;
		}

		// We now have two points (ax, cx) that straddle the minimum, and a
		// mid-point
		// bx with a value lower than either ay or cy.
		tx = ax
				+ (((bx - ax) * (bx - ax) * (cy - ay) - (cx - ax) * (cx - ax)
						* (by - ay)) / (2.0 * ((bx - ax) * (cy - ay) - (cx - ax)
						* (by - ay))));

		parameters.plusEqualsVectorTimes(line, tx - ox);
		optimizable.setParameters(parameters);
		logger.info("Ending cost = " + optimizable.computeLoss());

		// As a suggestion for the next initalStep, return the distance
		// from our initialStep to the minimum we found.
		return Math.max(1, tx - initialStep);
	}

}
