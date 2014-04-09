package com.dsi.parallax.ml.util.bounds;

public class LessThanOrEqualsValueBound extends UpperBound {

	private static final long serialVersionUID = -4630503304843144987L;

	public LessThanOrEqualsValueBound(double bound) {
		super(bound);
	}

	@Override
	public boolean satisfiesBound(double value) {
		return value <= bound;
	}

	@Override
	public String representBound() {
		if(null == representation)
			representation = "x <= " + bound;
		return representation;
	}
}
