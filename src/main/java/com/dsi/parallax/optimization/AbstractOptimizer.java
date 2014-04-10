/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import org.apache.log4j.Logger;

import com.dsi.parallax.ml.vector.LinearVector;

public abstract class AbstractOptimizer implements Optimizer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5378353121416141295L;
	protected int maxIterations = 10000;
	protected double tolerance = 0.00001;
	protected final double eps = 1.0e-5;
	protected double gradientTolerance = .0000001;
	protected ValueConvergenceType convergence = ValueConvergenceType.RELATIVE;
	protected final Logger logger = Logger.getLogger(this.getClass());
	
	protected boolean checkValueTerminationCondition(double value, double oldValue)
	{
		switch(convergence)
		{
		case FALSE:
			return false;
		case DIFFERENCE:
			return differenceValueConvergence(value, oldValue);
		case ABSOLUTE:
			return absoluteValueConvergence(value, oldValue);
		case RELATIVE:
		default:
			return relativeValueConvergence(value, oldValue);
		}
	}
	protected boolean relativeValueConvergence(double value, double oldValue)
	{
		return (2.0 * Math.abs(value - oldValue) <= tolerance
				* (Math.abs(value) + Math.abs(oldValue) + eps));
	}
	protected boolean absoluteValueConvergence(double value, double oldValue)
	{
		return (Math.abs(value - oldValue) <= tolerance);
	}
	protected boolean differenceValueConvergence(double value, double oldValue)
	{
		return (value<=tolerance);
	}
	protected boolean checkGradientTerminationCondition(LinearVector grad)
	{
		return grad.LPNorm(2) < gradientTolerance;
	}
	@Override
	public void setTolerance(double tolerance)
	{
		this.tolerance = tolerance;
	}
	@Override
	public void setGradTolerance(double gradTolerance)
	{
		this.gradientTolerance = gradTolerance;
	}
	@Override
	public void setValueConvergenceType(ValueConvergenceType type)
	{
		this.convergence = type;
	}
	
	@Override
	public boolean optimize() {
		return optimize(maxIterations);
	}
}
