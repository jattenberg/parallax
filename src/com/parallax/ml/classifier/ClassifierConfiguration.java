package com.parallax.ml.classifier;

import com.parallax.ml.util.option.Configuration;

/**
 * Base class for classifier Configurations; adds an extra requirement for
 * generating a classifier building with the stored parameter values
 * 
 * @param <C>
 *            the concrete type of classifier to be built
 * @param <B>
 *            the concrete type of classifier builder
 */
public abstract class ClassifierConfiguration<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
		extends Configuration<B> {

	/**
	 * Gets the classifier builder.
	 * 
	 * @return the classifier builder
	 */
	public abstract B getClassifierBuilder();
}
