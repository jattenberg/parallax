/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.vector.LinearVector;

public class NullTruncation extends AbstractGradientTruncation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5858140485391769264L;

	@Override
	public LinearVector truncateParameters(LinearVector vector) {
		// No truncation performed
		return vector;
	}

	@Override
	public void intialize() {
		// Nothing to do here!

	}

	@Override
	public TruncationType getType() {
		return TruncationType.NONE;
	}

}
