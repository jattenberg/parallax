/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.target.Target;

/**
 * class for determining how instances present at a node spilt up amongst the
 * child nodes. according to a specified utility function
 * 
 * @param <T>
 *            the type of label to be considered
 * @author jattenberg
 */
public interface Splitter<T extends Target> {

	/**
	 * build a preferred axis-aligned split on the supplied data. return the
	 * benefit of the determined split
	 * 
	 * @param criterion
	 *            the criterion used for determining the value of the split
	 * @param attributeValues
	 *            the attribute values to be assessed
	 * @param index
	 *            the index of the input space being split
	 * @return benefit of the induced split
	 */
	public Benefit buildSplit(SplitCriterion<T> criterion,
			AttributeValueSet<T> attributeValues, int index);

}
