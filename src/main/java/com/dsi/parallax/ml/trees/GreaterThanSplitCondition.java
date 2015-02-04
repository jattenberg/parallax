/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.dictionary.ReversableDictionary;
import com.dsi.parallax.ml.instance.Instance;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * {@link SplitCondition} which returns true if the input feature value is
 * greather than the split criteria
 * 
 * @author jattenberg
 */
public class GreaterThanSplitCondition extends SplitCondition {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8283058708390210632L;

	/**
	 * Instantiates a new greater than split condition.
	 * 
	 * @param index
	 *            dimension being queried by the split point
	 * @param splitValue
	 *            the split criteria
	 */
	public GreaterThanSplitCondition(int index, double splitValue) {
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
		return inst.getFeatureValue(index) > splitValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.trees.SplitCondition#representInequality()
	 */
	@Override
	public String representInequality() {
		return "x[" + index + "] > " + splitValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.trees.SplitCondition#naturalOrder()
	 */
	@Override
	public int naturalOrder() {
		return 1;
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
		return name + " > " + splitValue;
	}
}
