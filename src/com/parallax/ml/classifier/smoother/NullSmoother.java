/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.util.Collection;

import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * A dummy smoother; simply transforms the classifiers raw output score to
 * the range [0, 1].
 */
public class NullSmoother extends
		AbstractUpdateableSmoother<NullSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;

	public static NullSmoother SMOOTHER = new NullSmoother();

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		return prediction > 1 ? 1 : prediction < 0 ? 0 : prediction;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.UpdateableSmoother#update(com
	 * .parallax.ml.util.pair.PrimitivePair)
	 */
	@Override
	public void update(PrimitivePair p) {
		// Nothing to do!

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#train(java.util.Collection
	 * )
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		// Nothing to do!
	}

}
