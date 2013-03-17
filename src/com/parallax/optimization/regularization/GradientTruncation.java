/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.regularization;

import com.parallax.ml.vector.LinearVector;



public interface GradientTruncation
{
	public LinearVector truncateParameters(LinearVector vector);
	public void intialize();
	public TruncationType getType();
}
