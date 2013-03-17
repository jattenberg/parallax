/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.regularization;

import java.io.Serializable;

/**
 * worker class for gradient truncation classes
 * @author josh
 *
 */
public class GradientTruncationWorker implements Serializable
{

	private static final long serialVersionUID = 6131465723067879473L;
	private double threshold = 1.;
	private double alpha = 1.;
	private int bins = (int)Math.pow(2,16);
	private int period = 100;
	
	public double getThreshold()
	{
		return threshold;
	}
	public void setThreshold(double threshold)
	{
		this.threshold = threshold;
	}
	public double getAlpha()
	{
		return alpha;
	}
	public void setAlpha(double alpha)
	{
		this.alpha = alpha;
	}
	public int getBins()
	{
		return bins;
	}
	public void setBins(int bins)
	{
		this.bins = bins;
	}
	public int getPeriod()
	{
		return period;
	}
	public void setPeriod(int period)
	{
		this.period = period;
	}
	
	
}
