/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.regularization;

import java.io.Serializable;

public abstract class AbstractGradientTruncation implements GradientTruncation,
		Serializable {

	private static final long serialVersionUID = -1299510639407405783L;
	protected int period = 100;
	protected int epoch;
	protected int bins;
	protected int[] lastAccessed;
	private double theta = 1.;
	protected double alpha = 1.;

	public int getPeriod() {
		return period;
	}

	public void setPeriod(int period) {
		this.period = period;
	}

	public int getEpoch() {
		return epoch;
	}

	public void setEpoch(int epoch) {
		this.epoch = epoch;
	}

	public int getBins() {
		return bins;
	}

	public void setBins(int bins) {
		this.bins = bins;
	}

	@Override
	public void intialize() {
		lastAccessed = new int[bins];
	}

	public double getTheta() {
		return theta;
	}

	public void setTheta(double theta) {
		if (theta < 0)
			throw new IllegalArgumentException(
					"theta must be non-negative, value: " + theta);
		this.theta = theta;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpah) {
		this.alpha = alpah;
	}

}
