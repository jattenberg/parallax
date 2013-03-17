/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

/**
 * Container for storing the benefit of a particular split. 
 * Stores the index of the feature being split, and the numerical 
 * utility value of that split. 
 */
public class Benefit implements Comparable<Benefit> {

	/** The utility value of a particular split. */
	private final double utility;
	
	/** The index of the feature being split. */
	private final int index;
	
	/** The feature value to split on. */
	private double split;
	
	/**
	 * Instantiates a new benefit.
	 *
	 * @param index index of the feature being split.
	 * @param utility The utility value of a particular split.
	 */
	public Benefit(int index, double utility) {
		this.index = index;
		this.utility = utility;
	}
	
	/**
	 * Instantiates a new benefit.
	 *
	 * @param index index of the feature being split.
	 * @param utility The utility value of a particular split.
	 * @param split The feature value to split on.
	 */
	public Benefit(int index, double utility, double split) {
		this(index, utility);
		this.split = split;
	}
	
	/**
	 * Gets the utility, the value of a particular split.
	 *
	 * @return the utility
	 */
	public double getUtility() {
		return utility;
	}

	/**
	 * Gets the index of the feature being split on.
	 *
	 * @return the index
	 */
	public int getIndex() {
		return index;
	}

	/**
	 * Gets the value of the feature used to split.
	 *
	 * @return the split
	 */
	public double getSplit() {
		return split;
	}

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(Benefit o) {
		return -1 * Double.compare(this.utility, o.utility);
	}

}
