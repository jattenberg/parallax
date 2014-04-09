/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.smoother;

import java.util.Collection;

import com.dsi.parallax.ml.util.pair.PrimitivePair;

/**
 * Base class for smoothers- classes that transform raw classifier scores
 * into probability estimates- that can incorporate new training information
 * incrementally.
 * 
 * @author jattenberg
 */
public abstract class AbstractUpdateableSmoother<U extends AbstractUpdateableSmoother<U>>
		extends AbstractSmoother<U> implements UpdateableSmoother<U> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -4466573119115902662L;

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.AbstractSmoother#train(java
	 * .util.Collection)
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		for (PrimitivePair p : input) {
			update(p);
		}
	}
}
