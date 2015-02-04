/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.dictionary;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Maps;

import java.util.Collection;
import java.util.Map;

/**
 * BiMapReversableDictionary
 * 
 * @author Josh Attenberg
 */
public class BiMapReversableDictionary extends AbstractDictionary implements
		ReversableDictionary {

	private static final long serialVersionUID = 7216842702127214070L;
	private final Map<String, Collection<String>> DUMMYMAP = Maps.newHashMap();
	private final String DUMMYKEY = "key";
	private BiMap<String, Integer> dictionary;
	private int current;

	/**
	 * Class constructor specifying dimension to create
	 * 
	 * @param dimension
	 *            dimension
	 */
	public BiMapReversableDictionary(int dimension) {
		super(dimension);
		dictionary = HashBiMap.create();
		current = 0;
	}

	/**
	 * Class constructor specifying dimension and binary Feature to create
	 * 
	 * @param dimension
	 *            dimension
	 * @param binaryFeatures
	 *            binary Features
	 */
	public BiMapReversableDictionary(int dimension, boolean binaryFeatures) {
		this(dimension);
		this.binaryFeatures = binaryFeatures;
	}

	/**
	 * The method gets current size
	 * 
	 * @return int current size
	 */
	@Override
	public int getCurrentSize() {
		return current;
	}

	/**
	 * The method creates LinearVector by multiple text
	 * 
	 * @param text
	 *            text
	 * @return LinearVector
	 */
	@Override
	public LinearVector vectorFromText(Collection<String> text) {
		DUMMYMAP.put(DUMMYKEY, text);
		return vectorFromNamespacedText(DUMMYMAP, false);
	}

	/**
	 * The method creates LinearVector by multiple namespace text and namespace
	 * flag
	 * 
	 * @param namespacedText
	 *            namespace text
	 * @param namespace
	 *            namespace flag
	 * @return LinearVector
	 */
	@Override
	public LinearVector vectorFromNamespacedText(
			Map<String, Collection<String>> namespacedText, boolean namespace) {
		LinearVector vector = LinearVectorFactory.getVector(dimension);
		for (String ns : namespacedText.keySet()) {
			Collection<String> tokens = namespacedText.get(ns);
			for (String token : tokens) {
				String mapToken = namespace ? (ns + "_" + token) : token;
				int index = getOrAddFeature(mapToken);
				if (index < 0)
					continue;
				if (binaryFeatures)
					vector.resetValue(index, 1);
				else
					vector.updateValue(index, 1);
			}
		}
		return vector;
	}

	/**
	 * The method gets feature from dictionary or adds new feature to dictionary
	 * 
	 * @param feature
	 *            feature name
	 * @return int feature
	 */
	@Override
	public int getOrAddFeature(String feature) {
		if (!dictionary.containsKey(feature))
			if (current < dimension)
				dictionary.put(feature, current++);
			else
				return -1;
		return dictionary.get(feature);
	}

	/**
	 * The method gets feature by index of bitmap
	 * 
	 * @param index
	 *            index of bitmap
	 * @return String Feature
	 */
	@Override
	public String getFeatureFromIndex(int index) {
		return dictionary.inverse().get(index);
	}

	@Override
	public int dimensionFromString(String input) {
		if (!dictionary.containsKey(input)) {
			return -1;
		} else {
			return dictionary.get(input);
		}

	}
}
