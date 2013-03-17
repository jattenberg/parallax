/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

/**
 * Interface for AttributeValueCaches used to potentially speed up decision tree
 * computation. More precicely, this is a cache for AttributeValueLabel, storing
 * the distribution of attribute values and labels  in order
 * to potentially avoid re-sorting examples in the tree Map.
 * 
 * @author jattenberg
 */
public interface AttributeValueCache {

	/**
	 * Gets the attribute value set.
	 * 
	 * @param <T>
	 *           The type of label being predicted. 
	 * @param <I>
	 *            The type of instances input
	 * @param instances
	 *            the training data
	 * @param dimension
	 *            the dimension where a split is requested.
	 * @return the attribute value set
	 */
	public <T extends Target, I extends Instances<? extends Instance<T>>> AttributeValueSet<T> getAttributeValueSet(
			I instances, int dimension);
}
