package com.dsi.parallax.ml.util.bounds;

import java.io.Serializable;

public abstract class ValueBound implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2941328307290829232L;
	protected final double bound;
	protected String representation;

	protected ValueBound(double bound) {
		this.bound = bound;
	}

	public abstract boolean satisfiesBound(double value);
	public abstract String representBound();
	
	public double getBoundValue() {
		return bound;
	}
}
