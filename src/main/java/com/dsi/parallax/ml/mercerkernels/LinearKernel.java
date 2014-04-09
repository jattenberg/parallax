/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.mercerkernels;

import com.dsi.parallax.ml.instance.Instance;

/**
 * Linear Kernel; vanilla inner product
 */
public class LinearKernel implements Kernel {

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.mercerkernels.Kernel#InnerProduct(com.parallax.ml.instance
	 * .Instanze, com.parallax.ml.instance.Instanze)
	 */
	public double InnerProduct(Instance<?> x, Instance<?> y) {
		return x.dot(y) + 1d;
	}

}
