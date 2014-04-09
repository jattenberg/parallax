/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import java.util.TreeMap;

import com.dsi.parallax.ml.target.Target;

/**
 * represents ordered map of frequencies for each attribute value / target value
 * pair. Used for compressing label distributions for attribute value pairs in
 * order to speed up decision tree computation
 * 
 * @param <T>
 *            the type of label being considered.
 * @author jattenberg
 */
public class AttributeValueSet<T extends Target> extends
		TreeMap<AttributeValueLabel<T>, Double> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8979569288924259281L;

	/**
	 * Add attribute value label information to the set, where it it contributes
	 * to the distribution of label info.
	 * 
	 * @param attribute
	 *            the attribute being added.
	 * @return the number of the same attribute value with the same label
	 */
	public Double add(AttributeValueLabel<T> attribute) {
		if (containsKey(attribute))
			return put(attribute, get(attribute) + 1);
		else
			return put(attribute, 1.);
	}
}
