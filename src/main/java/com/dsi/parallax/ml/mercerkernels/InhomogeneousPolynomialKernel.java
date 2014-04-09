/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.mercerkernels;

import com.dsi.parallax.ml.instance.Instance;

/**
 * A kernel mapping with a inhomogeneous polynomial, (x*y + 1) ^ n
 */
public class InhomogeneousPolynomialKernel implements Kernel {

	/** The degree. */
	private final double degree;

	/**
	 * Instantiates a new inhomogeneous polynomial kernel.
	 * 
	 * @param degree
	 *            the degree
	 */
	public InhomogeneousPolynomialKernel(double degree) {
		this.degree = degree;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.mercerkernels.Kernel#InnerProduct(com.parallax.ml.instance
	 * .Instanze, com.parallax.ml.instance.Instanze)
	 */
	public double InnerProduct(Instance<?> x, Instance<?> y) {
		double ip = 0.0;
		for (int i = 0; i < x.getDimension(); i++) {
			ip += Math.pow(x.getFeatureValue(i) * y.getFeatureValue(i) + 1.0,
					degree);
		}
		return ip;
	}

	/**
	 * Gets the degree of the polynomial.
	 * 
	 * @return the degree of the polynomial
	 */
	public double getDegree() {
		return degree;
	}
}
