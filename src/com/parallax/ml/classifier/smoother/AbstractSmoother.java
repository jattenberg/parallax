/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.io.Serializable;
import java.util.Collection;

/**
 * Base class for Smoothers; classes that transform raw classifier scores
 * into probability estimates. Provides a default implementation of
 * {@link #train(Collection)} that does nothing.
 * 
 * @author jattenberg
 */
public abstract class AbstractSmoother<U extends AbstractSmoother<U>> implements Smoother<U>, Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1914389891062893195L;

	/** has training been performed? */
	protected boolean trained = false;

}
