/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

public class DummyTargetParser<T extends Target> implements TargetParser<T>{

	@Override
	public T parseTarget(String stringValue) {
		return null;
	}

}
