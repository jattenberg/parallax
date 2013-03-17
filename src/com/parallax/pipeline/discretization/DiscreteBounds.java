/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.discretization;

import java.util.Arrays;

/**
 * used when discretizing a continuous feature. stores boundaries between
 * different discrete buckets, and is able to map a continuous value to the
 * appropriate bucket
 * 
 * @author jattenberg
 * 
 */
public class DiscreteBounds {

	/** The bounds where input scores are broken up */
	private final double[] bounds;

	/**
	 * The starting point; usually discretization is done in the context of a
	 * larger set of feature / value pairs this means, the output discrete value
	 * must be inserted somewhere into a vector space, usually as a continuous
	 * range value, this is the starting index of that range
	 */
	private final int startingPoint;

	/**
	 * Instantiates a new discrete bounds.
	 * 
	 * @param startingPoint
	 *            The starting point; usually discretization is done in the
	 *            context of a larger set of feature / value pairs this means,
	 *            the output discrete value must be inserted somewhere into a
	 *            vector space, usually as a continuous range value, this is the
	 *            starting index of that range
	 * @param bounds
	 *            The bounds where input scores are broken up
	 */
	public DiscreteBounds(int startingPoint, double[] bounds) {
		this.bounds = bounds;
		this.startingPoint = startingPoint;
	}

	/**
	 * Gets The bounds where input scores are broken up
	 * 
	 * @return The bounds where input scores are broken up
	 */
	public double[] getBounds() {
		return Arrays.copyOf(bounds, bounds.length);
	}

	/**
	 * Gets The starting point; usually discretization is done in the context of
	 * a larger set of feature / value pairs this means, the output discrete
	 * value must be inserted somewhere into a vector space, usually as a
	 * continuous range value, this is the starting index of that range
	 * 
	 * @return the starting point
	 */
	public int getStartingPoint() {
		return startingPoint;
	}

	/**
	 * Discretize a continuous value into a categorical value
	 * 
	 * @param continuous
	 *            the continuous
	 * @return the int
	 */
	public int discretize(double continuous) {
		if (continuous < bounds[0])
			return 0;
		else if (continuous >= bounds[bounds.length - 1])
			return bounds.length - 1;
		else
			for (int i = 1; i < bounds.length; i++)
				if (continuous >= bounds[i - 1] && continuous < bounds[i])
					return i;

		throw new ArrayIndexOutOfBoundsException(continuous
				+ " seems to be out of bounds: " + Arrays.toString(bounds));
	}
}
