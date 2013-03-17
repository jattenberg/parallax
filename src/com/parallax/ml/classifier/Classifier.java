/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier;

import com.parallax.ml.model.Model;
import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * The Interface Classifier, describing all models that predict
 * {@link BinaryClassificationTarget} Methods are the same as defined by
 * {@link Model} with the above target
 * 
 * @param <C>
 *            The type of classifier, used for method chaining.
 */
public interface Classifier<C extends Classifier<C>> extends
		Model<BinaryClassificationTarget, C> {
	//Nothing to do here. 
}
