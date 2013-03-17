package com.parallax.ml.util.bounds;

public class LessThanValueBound extends UpperBound {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5928716763871002565L;
	
	public LessThanValueBound(double bound) {
		super(bound);
	}

	@Override
	public boolean satisfiesBound(double value) {
		return value < bound;
	}

	@Override
	public String representBound() {
		if(null == representation)
			representation = "x < " + bound;
		return representation;
	}

}
