/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static com.google.common.base.Preconditions.checkArgument;

import java.io.Serializable;

import com.parallax.ml.dictionary.ReversableDictionary;
import com.parallax.ml.instance.Instance;

/**
 * Base class for split conditions in decision trees
 */
public abstract class SplitCondition implements Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -804309210831577182L;

	/** The index of the input space that is considered for splits */
	protected final int index;

	/** The value used to determine the split criterion */
	protected final double splitValue;

	/**
	 * Instantiates a new split condition.
	 * 
	 * @param index
	 *            The index of the input space that is considered for splits
	 * @param splitValue
	 *            The value used to determine the split criterion
	 */
	protected SplitCondition(int index, double splitValue) {
		checkArgument(index >= 0, "index must be non-negative, given: %s",
				index);
		this.index = index;
		this.splitValue = splitValue;
	}

	/**
	 * Gets the split index; The index of the input space that is considered for
	 * splits
	 * 
	 * @return The index of the input space that is considered for splits
	 */
	public int getSplitIndex() {
		return index;
	}

	/**
	 * Gets the split value; The value used to determine the split criterion
	 * 
	 * @return The value used to determine the split criterion
	 */
	public double getSplitValue() {
		return splitValue;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		return representInequality();
	}

	/**
	 * Does the input instance satisfy the split condition, used to decide which
	 * branch to descend.
	 * 
	 * @param inst
	 *            the instance considered for the split
	 * @return true, if successful
	 */
	public abstract boolean satisfiesSplit(Instance<?> inst);

	/**
	 * Represent the inequality as a string.
	 * 
	 * @return the string representing the inequality
	 */
	public abstract String representInequality();

	/**
	 * Represent inequality using a reversable dictionary to infer the feature
	 * name
	 * 
	 * @param dict
	 *            the dictionary used to supply the "names" of features
	 * @return the string representing the inequality
	 */
	public abstract String representInequality(ReversableDictionary dict);

	/**
	 * Natural order.
	 * 
	 * @return the int
	 */
	public abstract int naturalOrder();
}
