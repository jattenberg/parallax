/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.dictionary;

import java.util.Collection;

import com.parallax.ml.vector.LinearVector;

/**
 * ReversableDictionary is interface to define create Linear Vector and how to get feature
 *
 * @author Josh Attenberg
 */
public interface ReversableDictionary extends Dictionary {
	public int getCurrentSize();
	LinearVector vectorFromText(Collection<String> text);
	public int getOrAddFeature(String feature);
	public String getFeatureFromIndex(int index);
}
