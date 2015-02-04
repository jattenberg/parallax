/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.dictionary;

import com.dsi.parallax.ml.vector.LinearVector;

import java.util.Collection;

/**
 * ReversableDictionary is interface to define create Linear Vector and how to get feature
 *
 * @author Josh Attenberg
 */
public interface ReversableDictionary extends Dictionary {
	public int getCurrentSize();
	@Override
	LinearVector vectorFromText(Collection<String> text);
	public int getOrAddFeature(String feature);
	public String getFeatureFromIndex(int index);
}
