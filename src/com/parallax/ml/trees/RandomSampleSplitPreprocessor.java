/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * The Class RandomSampleSplitPreprocessor.
 */
public class RandomSampleSplitPreprocessor implements SplitPreProcessor {

	/** The percentage. */
	private final double percentage;
	
	/**
	 * Instantiates a new random sample split preprocessor.
	 *
	 * @param percentage the percentage
	 */
	public RandomSampleSplitPreprocessor(double percentage) {
		this.percentage = percentage;
	}
	
	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.SplitPreProcessor#preprocess(com.parallax.ml.instance.Instances)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public <T extends Target, I extends Instances<? extends Instance<T>>> I preprocess(
			I instances) {
		return (I)instances.getBag(percentage).getBagInstances();
	}

}
