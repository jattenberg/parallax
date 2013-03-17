/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import java.util.Set;

import com.google.common.collect.Sets;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

/**
 * Terminator that halts further splits in a decision tree if only a single
 * label type exists in a node
 * 
 * @param <T>
 *            the type of label considered
 */
public class SingleLabelTerminator<T extends Target> implements Terminator<T> {

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.Terminator#terminate(com.parallax.ml.instance.Instances
	 * , int)
	 */
	@Override
	public <I extends Instances<? extends Instance<T>>> boolean terminate(
			I instances, int depth) {
		Set<T> labelSet = Sets.newTreeSet();

		for (Instance<T> inst : instances) {
			labelSet.add(inst.getLabel());
			if (labelSet.size() > 1)
				return false;
		}
		return true;
	}

}
