/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.mercerkernels;

import com.parallax.ml.instance.Instance;

/**
 * The Interface for mercer kernel functions describing the inner product for
 * vectors in alternative spaces
 */
public interface Kernel {

	/**
	 * Inner product.
	 * 
	 * @param x
	 *            the x
	 * @param y
	 *            the y
	 * @return the double
	 */
	public double InnerProduct(Instance<?> x, Instance<?> y);
}
