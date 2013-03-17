/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.kernelmethods;

import com.parallax.ml.classifier.AbstractUpdateableClassifier;
import com.parallax.ml.mercerkernels.Kernel;
import com.parallax.ml.mercerkernels.PolynomialKernel;

/**
 * base case for updateable classifiers that utilize mercer kernels to 
 * compute inner products in higher dimensional spaces. These classifiers can 
 * incrementally incorporate additional training data as it becomes available. 
 *
 * @param <C> the concrete type of the classifier itself used for method chaining.
 * @author jattenberg
 */
public abstract class AbstractUpdateableKernelClassifier<C extends AbstractUpdateableKernelClassifier<C>>
		extends AbstractUpdateableClassifier<C> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 4688195832759197484L;
	
	/** The kernel used to compute inner products in the kernel method. */
	protected Kernel kernel = new PolynomialKernel(2.);
	
	/** The model itself. used for method chaining. */
	protected C model;

	/**
	 * Instantiates a new abstract updateable kernel classifier.
	 *
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected AbstractUpdateableKernelClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		model = getModel();
	}

	/**
	 * Sets the kernel used for computing inner products in higher dimensional spaces
	 *
	 * @param kernel the kernel
	 * @return the model itself, used for method chaining. 
	 */
	public C setKernel(Kernel kernel) {
		this.kernel = kernel;
		return model;
	}
}
