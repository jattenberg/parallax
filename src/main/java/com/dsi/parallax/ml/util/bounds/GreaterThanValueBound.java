package com.dsi.parallax.ml.util.bounds;

public class GreaterThanValueBound extends LowerBound {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2882739525380245025L;

	public GreaterThanValueBound(double bound) {
		super(bound);
	}

	@Override
	public boolean satisfiesBound(double value) {
		return value > bound;
	}

	@Override
	public String representBound() {
		if(null == representation)
			representation = "x > " + bound;
		return representation;
	}

}
