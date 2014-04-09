/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.target.Target;

/**
 * class establishing the objective value of a particular split in a decision
 * tree. for instance, info gain or gini
 * 
 * @param <T>
 *            the type of target considered in the decision tree
 * @author jattenberg
 */
public interface SplitCriterion<T extends Target> {

	/**
	 * Compute objective the objective of a particular split on the supplied
	 * attribute values.
	 * 
	 * @param attributeValues
	 *            set of attribute values present at a node
	 * @param split
	 *            value where examples are split
	 * @return the objective of the proposed split
	 */
	public double computeObjective(AttributeValueSet<T> attributeValues,
			double split);
}
