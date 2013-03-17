/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.vector.LinearVector;

/**
 * DummyProjection simply projects incoming vectors with the identity matrix; no
 * projection at all
 */
public class DummyProjection extends AbstractProjection {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6605838218748611268L;

	/**
	 * Instantiates a new dummy projection.
	 * 
	 * @param inputDimension
	 *            the dimension of the input space
	 */
	public DummyProjection(int inputDimension) {
		super(inputDimension, inputDimension);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.projection.Projection#project(com.parallax.ml.util.vector
	 * .LinearVector)
	 */
	@Override
	public LinearVector project(LinearVector x) {
		return x;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.projection.AbstractProjection#project(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<?>> I project(I x) {
		return x;
	}
}
