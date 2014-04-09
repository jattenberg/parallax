/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import com.dsi.parallax.ml.vector.LinearVector;

/**
 * Optimizable functions must implement these methods; these are the methods
 * used by the optimization 
 * @author jattenberg
 *
 */
public interface Optimizable {
	public int getNumParameters();

	public LinearVector getVector();

	public double getParameter(int index);
	
	public void setParameter(int index, double value);

	public void setParameters(LinearVector params);

	public Gradient computeGradient();
	
	public Gradient computeGradient(LinearVector params);
	
	public double computeLoss();

	public double computeLoss(LinearVector params);
}
