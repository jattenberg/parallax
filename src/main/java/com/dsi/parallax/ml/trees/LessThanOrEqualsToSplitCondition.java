/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.dictionary.ReversableDictionary;
import com.dsi.parallax.ml.instance.Instance;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Split condition that is true only if the value being queried is less than or
 * equal to the fixed split value
 */
public class LessThanOrEqualsToSplitCondition extends SplitCondition {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8283058708390210632L;

	/**
	 * Instantiates a new less than or equals to split condition.
	 * 
	 * @param index
	 *            the index of the dimension being split upon
	 * @param splitValue
	 *            the value to which feature values are compared
	 */
	public LessThanOrEqualsToSplitCondition(int index, double splitValue) {
		super(index, splitValue);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.SplitCondition#satisfiesSplit(com.parallax.ml.instance
	 * .Instanze)
	 */
	@Override
	public boolean satisfiesSplit(Instance<?> inst) {
		checkArgument(
				inst.getDimension() > index,
				"instance not large enough (%s dimensions) to be split at index: %s",
				inst.getDimension(), index);
		return inst.getFeatureValue(index) <= splitValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.trees.SplitCondition#representInequality()
	 */
	@Override
	public String representInequality() {
		return "x[" + index + "] <= " + splitValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.trees.SplitCondition#naturalOrder()
	 */
	@Override
	public int naturalOrder() {
		return -1;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.SplitCondition#representInequality(com.parallax
	 * .ml.dictionary.ReversableDictionary)
	 */
	@Override
	public String representInequality(ReversableDictionary dict) {
		String name = dict.getFeatureFromIndex(index);
		return name + " <= " + splitValue;
	}

}
