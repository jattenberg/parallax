/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.mercerkernels;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.util.MLUtils;

/**
 * Gaussian Radial Basis Function Mercer Kernel.
 * {@link <a href="http://www.scholarpedia.org/article/Radial_basis_function#Examples_of_radial_basis_functions">Radial Basis Function</a>}
 * for more info
 * 
 * @author jattenberg
 */
public class GaussianRBFKernel implements Kernel {

	/** the variance parameter used in the rbf kernel. */
	private final double sigma;

	/**
	 * Instantiates a new gaussian rbf kernel.
	 * 
	 * @param sigma
	 *            variance parameter used in the rbf kernel
	 */
	public GaussianRBFKernel(double sigma) {
		this.sigma = sigma;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.mercerkernels.Kernel#InnerProduct(com.parallax.ml.instance
	 * .Instanze, com.parallax.ml.instance.Instanze)
	 */
	@Override
	public double InnerProduct(Instance<?> x, Instance<?> y) {
		double ip = Math.pow(MLUtils.euclidianDistance(x, y), 2);
		ip *= -sigma;
		return Math.exp(ip);

	}

	/**
	 * Gets the sigma, variance parameter used in the rbf kernel
	 * 
	 * @return the sigma, variance parameter used in the rbf kernel
	 */
	public double getSigma() {
		return sigma;
	}

}
