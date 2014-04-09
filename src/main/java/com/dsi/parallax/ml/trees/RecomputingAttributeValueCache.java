/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * The Class RecomputingAttributeValueCache.
 */
public class RecomputingAttributeValueCache implements AttributeValueCache {

	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.AttributeValueCache#getAttributeValueSet(com.parallax.ml.instance.Instances, int)
	 */
	@Override
	public <T extends Target, I extends Instances<? extends Instance<T>>> AttributeValueSet<T> getAttributeValueSet(
			I instances, int dimension) {
		AttributeValueSet<T> set = new AttributeValueSet<T>();
		for (Instance<T> inst : instances) {
			set.add(new AttributeValueLabel<T>(inst.getFeatureValue(dimension),
					inst.getLabel()));
		}
		return set;
	}

}
