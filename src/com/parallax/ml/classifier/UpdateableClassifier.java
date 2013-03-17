/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * The Interface UpdateableClassifier, describing methods used by all classifiers
 * that are incrementally updateable. That is, models that may incorporate new 
 * labeled information as it becomes available as opposed to training on a large
 * batch of information all at once. 
 *
 * @param <C>
 *            The type of updateable classifier, used for method chaining.
 */
public interface UpdateableClassifier< C extends UpdateableClassifier<C>> extends Classifier<C>
{

	/**
	 * Update- incorporate information contained in a single labeled example. 
	 *
	 * @param <I> the type of {@link Instance}} to be considered
	 * @param instst Instanze used for training. 
	 */
	public <I extends Instance<BinaryClassificationTarget>> void  update (I  instst);
	
	/**
	 * Update- incorporate information contained in a group of labeled information
	 *
	 * @param <I> the type of {@link Instance}} to be considered
	 * @param insts Collection of Instanze used for training. 
	 */
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void update (I insts);
	
}
