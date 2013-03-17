/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.mercerkernels;

import java.util.HashSet;
import java.util.Set;

import com.parallax.ml.instance.Instance;

// TODO: Auto-generated Javadoc
/**
 * The Class RBFKernel.
 */
public class RBFKernel implements Kernel {

	/** The gamma. */
	private final double gamma;

	/**
	 * Instantiates a new rBF kernel.
	 *
	 * @param gamma the gamma
	 */
	public RBFKernel(double gamma) {
		this.gamma = gamma;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.mercerkernels.Kernel#InnerProduct(com.parallax.ml.instance.Instanze, com.parallax.ml.instance.Instanze)
	 */
	public double InnerProduct(Instance<?> x, Instance<?> y) {
		double ip = 0.0;
		Set<Integer> dims = new HashSet<Integer>();
		for (int x_i : x) {
			dims.add(x_i);
			ip += Math.pow(x.getFeatureValue(x_i) - y.getFeatureValue(x_i), 2);
		}
		for (int x_i : y) {
			if (!dims.contains(x_i))
				ip += Math.pow(x.getFeatureValue(x_i) - y.getFeatureValue(x_i),
						2);

		}

		ip *= gamma;

		return Math.exp(-ip);
	}

	/**
	 * Gets the gamma.
	 *
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma;
	}

}
