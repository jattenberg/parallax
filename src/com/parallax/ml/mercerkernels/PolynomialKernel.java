/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.mercerkernels;

import com.parallax.ml.instance.Instance;

// TODO: Auto-generated Javadoc
/**
 * The Class PolynomialKernel.
 */
public class PolynomialKernel implements Kernel {

	/** The degree. */
	private final double degree;

	/**
	 * Instantiates a new polynomial kernel.
	 *
	 * @param degree the degree
	 */
	public PolynomialKernel(double degree) {
		this.degree = degree;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.mercerkernels.Kernel#InnerProduct(com.parallax.ml.instance.Instanze, com.parallax.ml.instance.Instanze)
	 */
	public double InnerProduct(Instance<?> x, Instance<?> y) {
		double ip = 0.0;
		for (int x_i : y) {
			double y_i = y.getFeatureValue(x_i);
			double z_i = x.getFeatureValue(x_i);
			ip += Math.pow(z_i * y_i, degree);
		}

		return ip;
	}

	/**
	 * Gets the degree.
	 *
	 * @return the degree
	 */
	public double getDegree() {
		return degree;
	}

}
