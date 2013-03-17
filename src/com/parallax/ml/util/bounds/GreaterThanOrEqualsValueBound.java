package com.parallax.ml.util.bounds;

public class GreaterThanOrEqualsValueBound extends LowerBound {

	private static final long serialVersionUID = 3527044487347051200L;

	public GreaterThanOrEqualsValueBound(double bound) {
		super(bound);
	}

	@Override
	public boolean satisfiesBound(double value) {
		return value >= bound;
	}

	@Override
	public String representBound() {
		if(null == representation)
			representation = "x >= " + bound;
		return representation;
	}
	
	
}
