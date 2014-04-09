/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

/**
 * TargetParser is the interface which all the label parser classes should implement
 * 
 * @author jattenberg
 *
 * @param <T>
 * 			the type of target being parsed (binary, multiclass etc)
 */
public interface TargetParser<T extends Target> {
	public T parseTarget(String stringValue);
}
