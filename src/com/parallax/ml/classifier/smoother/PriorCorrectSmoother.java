/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.util.Collection;

import org.apache.log4j.Logger;

import com.parallax.ml.classifier.Classifier;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * The Class PriorCorrectSmoother.
 */
public class PriorCorrectSmoother extends
		AbstractSmoother<PriorCorrectSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5340619628101591533L;

	/** The baserate. */
	private double baserate = 0.5;

	/** The oldrate. */
	private double oldrate = 0.5;

	/** The smoother. */
	private static NullSmoother smoother = new NullSmoother();

	/** The Constant LOGGER. */
	private static final Logger LOGGER = Logger
			.getLogger(PriorCorrectSmoother.class);

	/** The Constant EPSILON. */
	private static final double EPSILON = 0.00001;

	/**
	 * Instantiates a new prior correct smoother.
	 * 
	 * @param baserate
	 *            the baserate
	 * @param oldrate
	 *            the oldrate
	 */
	public PriorCorrectSmoother(double baserate, double oldrate) {
		this.baserate = baserate;
		this.oldrate = oldrate;
	}

	/**
	 * Instantiates a new prior correct smoother.
	 * 
	 * @param <C>
	 *            the generic type
	 * @param pool
	 *            the pool
	 * @param training
	 *            the training
	 * @param model
	 *            the model
	 * @param EM
	 *            the em
	 * @param discrete
	 *            the discrete
	 */
	public <C extends Classifier<C>> PriorCorrectSmoother(
			BinaryClassificationInstances pool,
			BinaryClassificationInstances training, C model, boolean EM,
			boolean discrete) {
		computeBaseRates(pool, training, model, EM, discrete);
	}

	/**
	 * Compute base rates.
	 * 
	 * @param <C>
	 *            the generic type
	 * @param pool
	 *            the pool
	 * @param training
	 *            the training
	 * @param model
	 *            the model
	 * @param EM
	 *            the em
	 * @param discrete
	 *            the discrete
	 */
	private <C extends Classifier<C>> void computeBaseRates(
			BinaryClassificationInstances pool,
			BinaryClassificationInstances training, C model, boolean EM,
			boolean discrete) {
		computeOldRate(training);

		EMbaserate(pool, model, EM, discrete);
	}

	/**
	 * E mbaserate.
	 * 
	 * @param <C>
	 *            the generic type
	 * @param pool
	 *            the pool
	 * @param model
	 *            the model
	 * @param EM
	 *            the em
	 * @param discrete
	 *            the discrete
	 */
	private <C extends Classifier<C>> void EMbaserate(
			BinaryClassificationInstances pool, C model, boolean EM,
			boolean discrete) {
		baserate = oldrate;
		double brold = baserate;
		do {
			brold = baserate;
			double tot = 0;
			for (Instance<BinaryClassificationTarget> x : pool) {
				double tmp = smooth(model.predict(x).getValue());
				tot += discrete ? (tmp > 0.5 ? 1 : 0) : tmp;
			}
			baserate = (tot + EPSILON) / ((double) pool.size() + 2 * EPSILON);
		} while (EM && Math.abs(baserate - brold) > EPSILON);
		LOGGER.info("baserate: " + baserate + ", training rate: " + oldrate
				+ " actual: " + (double) pool.getNumPos()
				/ (double) (pool.getNumPos() + pool.getNumNeg()));
	}

	/**
	 * Compute old rate.
	 * 
	 * @param training
	 *            the training
	 */
	private void computeOldRate(BinaryClassificationInstances training) {
		if (training.size() > 0) {
			oldrate = 0;
			for (Instance<BinaryClassificationTarget> x : training)
				oldrate += smoother.smooth(x.getLabel().getValue());
			oldrate = (oldrate + EPSILON)
					/ ((double) training.size() + 2 * EPSILON);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		double p = smoother.smooth(prediction);
		return baserate * (p - p * oldrate)
				/ (oldrate - p * oldrate + p * baserate - baserate * oldrate);
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
