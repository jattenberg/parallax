/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization;

import java.io.Serializable;

public interface Optimizer extends Serializable {
	public boolean optimize ();
	public boolean optimize (int numIterations);
	public boolean isConverged();
	public Optimizable getOptimizable();
	public void setValueConvergenceType(ValueConvergenceType type);
	public void setTolerance(double tolerance);
	public void setGradTolerance(double gradTolerance);
}
