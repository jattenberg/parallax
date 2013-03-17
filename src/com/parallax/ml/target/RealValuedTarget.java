/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.target;


/**
 * RealValuedTarget defines real value target
 * 
 * @author Josh Attenberg
 */
public class RealValuedTarget implements Target, Comparable<Target> {
	protected double value; // can this be final?

	/**
	 * Class constructor specifying real value to create
	 * 
	 * @param value
	 *            real value
	 */
	public RealValuedTarget(double value) {
		setValue(value);
	}

	/**
	 * The method gets real value
	 * 
	 * @return real value
	 */
	public double getValue() {
		return value;
	}

	/**
	 * The method sets real value
	 * 
	 * @param value
	 *            real value
	 */
	public void setValue(double value) {
		this.value = value;
	}

	@Override
	public String toString() {
		return "" + this.value;
	}

	@Override
	public int compareTo(Target o) {
		if (o instanceof RealValuedTarget
				|| o instanceof BinaryClassificationTarget) {
			RealValuedTarget r = (RealValuedTarget) o;
			return Double.compare(value, r.value);
		} else {
			return -1;
		}
	}

	@Override
	public boolean equals(Object o) {
		if (!o.getClass().isAssignableFrom(this.getClass())) {
			return false;
		} else {
			RealValuedTarget t = (RealValuedTarget) o;
			return value == t.value;
		}

	}
}
