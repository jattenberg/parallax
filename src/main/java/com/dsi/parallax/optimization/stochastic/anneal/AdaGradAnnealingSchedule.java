package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.optimization.Gradient;

/**
 * see Adaptive Subgradient Methods for Online Learning and Stochastic
 * Optimization, Duchi et al.
 * 
 * @author jattenberg
 * 
 */
public class AdaGradAnnealingSchedule extends AnnealingSchedule implements
		GradientUpdateable {

	/** The grad storage. */
	private transient double[] gradStorage;

	public AdaGradAnnealingSchedule(double initialRate) {
		super(initialRate);
	}

	/**
	 * Component wise eta.
	 * 
	 * @param dim
	 *            the dim
	 * @param grads
	 *            the grads
	 * @return the double
	 */
	@Override
	public double learningRate(int epoch, int dimension) {
		if (gradStorage == null) {
			return initialLearningRate;
		} else {
			return MLUtils.inverseSquareRoot(new Double(gradStorage[dimension])
					.floatValue());
		}
	}

	@Override
	public void update(Gradient grad) {
		if (gradStorage == null)
			gradStorage = new double[grad.size()];
		for (int x_i : grad) {
			gradStorage[x_i] += Math.pow(grad.getValue(x_i), 2d);
		}
	}
}
