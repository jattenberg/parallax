/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.util.Collection;
import java.util.EnumSet;

import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * The Interface Smoother.
 */
public interface Smoother<R extends Smoother<R>> {

	/** The Constant DUMMY_SMOOTHER. */
	public static final Smoother<?> DUMMY_SMOOTHER = new NullSmoother();

	/** The Constant UPDATEABLE_SMOOTHERS. */
	public static final EnumSet<SmootherType> UPDATEABLE_SMOOTHERS = EnumSet
			.of(SmootherType.NONE, SmootherType.UPDATEABLEPLATT);

	/**
	 * Regularize.
	 * 
	 * @param prediction
	 *            the prediction
	 * @return the double
	 */
	public double smooth(double prediction);

	/**
	 * Train.
	 * 
	 * @param input
	 *            the input
	 */
	public void train(Collection<PrimitivePair> input); // Pair = prediction,
														// label
}
